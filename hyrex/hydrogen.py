import numpy as np

import jax.numpy as jnp
from jax import config, lax, grad, vmap
import equinox as eqx

from diffrax import diffeqsolve, SaveAt, ODETerm, Tsit5, Kvaerno3, PIDController, DiscreteTerminatingEvent, ForwardMode, Event

from . import cosmology
from .cosmology import mH, c, hbar, kB
from . import recomb_functions
from .array_with_padding import array_with_padding
config.update("jax_enable_x64", True)

import os
file_dir = os.path.dirname(__file__)

class hydrogen_model(eqx.Module):
    """
    Hydrogen recombination model implementation.

    Computes hydrogen ionization fraction evolution through multiple phases:
    Saha equilibrium, post-Saha expansion, HYREC-2 EMLA with two-photon processes,
    and late-time EMLA-only evolution.

    Methods:
    --------
    get_hydrogen_history : Compute full hydrogen recombination history (units: dimensionless)
    Saha_equilibrium : Compute Saha equilibrium phase (units: dimensionless)
    post_Saha_expansion : Compute post-Saha expansion phase (units: dimensionless)
    solve_emla_twophoton : Solve EMLA with two-photon processes (units: dimensionless)
    solve_emla : Solve EMLA-only evolution (units: dimensionless)
    xe_derivative_twophoton : Compute ionization fraction derivative with two-photon (units: dimensionless)
    xe_tm_derivative : Compute coupled xe and Tm derivatives (units: dimensionless, eV)
    dxe_dlna_twophoton : Compute two-photon recombination rate (units: dimensionless)
    get_current_correction_func : Interpolate SWIFT correction function (units: dimensionless)
    steady_state_equations : Set up steady-state level population equations (units: dimensionless)
    Lyn_esc_rate : Compute Lyman escape rate (units: s^{-1})
    """
    integration_spacing : jnp.float64
    swift : jnp.array
    lna_axis_late : jnp.array
    concrete_axis_size : jnp.array

    xe_4He : jnp.array
    lna_4He : jnp.array

    last_4He_lna : jnp.float64
    twog_redshift : jnp.float64

    def __init__(self,xe_4He,lna_4He,lna_axis_late,last_4He_lna,twog_redshift,integration_spacing = 5.0e-4, Nsteps=800,swift = jnp.array(np.loadtxt(file_dir+"/tabs/fit_swift.dat"))):
        """
        Initialize hydrogen recombination model.

        Parameters:
        -----------
        xe_4He : array_with_padding
            Helium ionization fraction from previous calculation
        lna_4He : array_with_padding
            Log scale factor array from helium calculation
        lna_axis_late : array
            Late-time log scale factor grid for EMLA-only phase
        last_4He_lna : float
            Final log scale factor from helium recombination
        integration_spacing : float, optional
            Step size for integration (default: 5.0e-4)
        Nsteps : int, optional
            Maximum number of integration steps (default: 800)
        swift : array, optional
            SWIFT correction function tabulation
        """
        self.integration_spacing = integration_spacing
        self.swift = swift

        # Define time axes
        self.lna_axis_late = lna_axis_late
        self.concrete_axis_size = jnp.zeros(Nsteps)

        # pull in hydrogen
        self.xe_4He = xe_4He
        self.lna_4He = lna_4He

        self.last_4He_lna = last_4He_lna
        self.twog_redshift = twog_redshift

    def __call__(self, h, omega_b, omega_cdm, Neff, YHe, rtol=1e-6, atol=1e-9,solver=Kvaerno3(),max_steps=1024):
        """
        Compute hydrogen recombination history.

        Parameters:
        -----------
        h : float
            Hubble parameter
        omega_b : float
            The baryon density Omega_b h^2
        omega_cdm : float
            The density of Cold Dark Matter Omega_cdm h^2
        Neff : float
            Effective number of neutrinos
        YHe : float
            Helium fraction
        rtol : float, optional
            Relative tolerance for ODE solver (default: 1e-6)
        atol : float, optional
            Absolute tolerance for ODE solver (default: 1e-9)
        solver : diffrax.Solver, optional
            ODE solver instance (default: Kvaerno3())
        max_steps : int, optional
            Maximum solver steps (default: 1024)

        Returns:
        --------
        tuple
            (xe_full, lna_full, Tm, lna_Tm) - ionization fraction, log scale factor,
            matter temperature, and temperature grid
        """
        return self.get_hydrogen_history(h, omega_b, omega_cdm, Neff, YHe, rtol, atol, solver, max_steps)
    
    def get_hydrogen_history(self, h, omega_b, omega_cdm, Neff, YHe, rtol=1e-6, atol=1e-9,solver=Kvaerno3(),max_steps=1024):
        """
        Compute complete hydrogen recombination history through all phases.

        Sequentially computes hydrogen ionization fraction through post-Saha 
        expansion, HYREC-2 EMLA with two-photon processes, and late-time 
        EMLA-only evolution phases.

        Parameters:
        -----------
        h : float
            Hubble parameter
        omega_b : float
            The baryon density Omega_b h^2
        omega_cdm : float
            The density of Cold Dark Matter Omega_cdm h^2
        Neff : float
            Effective number of neutrinos
        YHe : float
            Helium fraction
        rtol : float, optional
            Relative tolerance for ODE solver (default: 1e-6)
        atol : float, optional
            Absolute tolerance for ODE solver (default: 1e-9)
        solver : diffrax.Solver, optional
            ODE solver instance (default: Kvaerno3())
        max_steps : int, optional
            Maximum solver steps (default: 1024)

        Returns:
        --------
        tuple
            (xe_full, lna_full, Tm, lna_Tm) containing ionization fraction evolution,
            log scale factor grid, matter temperature, and temperature grid
        """

        # Start computing xe at different phases
        ################## move to H ################## 

        ### POST SAHA EXPANSION PHASE ###
        xe_output_post, lna_output_post = self.post_Saha_expansion(self.last_4He_lna+self.integration_spacing, 
                                                                   h, omega_b, omega_cdm, Neff, YHe)
        
        xe_4He_and_post = self.xe_4He.concat(array_with_padding(xe_output_post))
        lna_4He_and_post = self.lna_4He.concat(array_with_padding(lna_output_post))

        ### END OF POST SAHA EXPANSION PHASE ###

        ### HYREC2 EMLA + FULL TWO PHOTON PHASE ###

        xe_output_2g, lna_output_2g = self.solve_emla_twophoton(lna_4He_and_post.lastval, -jnp.log(self.twog_redshift), 
                                                                xe_4He_and_post.lastval, h, omega_b, omega_cdm, Neff, YHe, 
                                                                rtol, atol, solver, max_steps)

        xe_4He_post_2g = xe_4He_and_post.concat(array_with_padding(xe_output_2g))
        lna_4He_post_2g = lna_4He_and_post.concat(array_with_padding(lna_output_2g))
        ### END HYREC2 EMLA + FULL TWO PHOTON PHASE ###

        ### HYREC2 EMLA ONLY PHASE ###

        xe_output_late, Tm_output_late, lna_output_late = self.solve_emla(self.lna_axis_late, xe_4He_post_2g.lastval, 
                                                         h, omega_b, omega_cdm, Neff, YHe, rtol, atol, solver)

        # lna_Tm = array_with_padding(self.lna_axis_late)
        lna_Tm = array_with_padding(lna_output_late)
        Tm = array_with_padding(Tm_output_late)

        xe_4He_post_2g_late = xe_4He_post_2g.concat(array_with_padding(xe_output_late))
        # lna_4He_post_2g_late = lna_4He_post_2g.concat(array_with_padding(self.lna_axis_late))
        lna_4He_post_2g_late = lna_4He_post_2g.concat(lna_Tm)
        ### END OF HYREC2 EMLA ONLY PHASE ###

        ### Begin TLA phase ###
        xe_output_TLA, Tm_output_TLA, lna_output_TLA = self.solve_TLA(lna_Tm.lastval, self.lna_axis_late, 
                                                                      xe_4He_post_2g_late.lastval, Tm.lastval, 
                                                                      h, omega_b, omega_cdm, Neff, YHe)

        xe_all = xe_4He_post_2g_late.concat(array_with_padding(xe_output_TLA))
        lna_all = lna_4He_post_2g_late.concat(array_with_padding(lna_output_TLA))
        Tm_all = Tm.concat(array_with_padding(Tm_output_TLA))
        lna_Tm_all = lna_Tm.concat(array_with_padding(lna_output_TLA))
        
        
        # return (xe_4He_post_2g_late, lna_4He_post_2g_late, Tm, lna_Tm)
        return (xe_all, lna_all, Tm_all, lna_Tm_all)


    def post_Saha_expansion(self, starting_lna, h, omega_b, omega_cdm, Neff, YHe, threshold=1e-5):
        """
        Compute post-Saha expansion phase with two-photon corrections.

        Calculates ionization fraction including two-photon processes as
        perturbative corrections to Saha equilibrium until deviations
        exceed threshold.

        Parameters:
        -----------
        starting_lna : float
            Initial log scale factor
        h : float
            Hubble parameter
        omega_b : float
            The baryon density Omega_b h^2
        omega_cdm : float
            The density of Cold Dark Matter Omega_cdm h^2
        Neff : float
            Effective number of neutrinos
        YHe : float
            Helium fraction
        threshold : float, optional
            Threshold for deviation from Saha (default: 1e-5)

        Returns:
        --------
        tuple
            (xe_output, lna_output) - ionization fraction and log scale factor arrays
        """
        # Calculate omega_rad today using input Neff.
        omega_rad = cosmology.omega_rad0(Neff)  

        # Initial conditions
        z0_local = jnp.exp(-starting_lna) - 1.
        TCMB = cosmology.TCMB(z0_local)
        nH = cosmology.nH(z0_local, omega_b, YHe)
        xe0, _ = recomb_functions.xe_Saha(TCMB, nH)  # Assume initially in Saha equilibrium

        # Pre-allocate xe_output 
        xe_output = jnp.ones_like(self.concrete_axis_size)*jnp.inf
        lna_output = jnp.ones_like(self.concrete_axis_size)*jnp.inf
        iz = 0
        xe = xe0
        stop = False

        def compute_xe(carry):
            xe_output, lna_output, xe, iz, stop = carry

            lna = starting_lna + iz*self.integration_spacing
            z = jnp.exp(-lna) - 1.

            # Cosmological parameters
            TCMB = cosmology.TCMB(z)
            nH = cosmology.nH(z, omega_b, YHe)
            H = cosmology.Hubble(z, h, omega_b, omega_cdm, omega_rad)

            # Saha equilibrium for xe
            xe_Saha, s = recomb_functions.xe_Saha(TCMB, nH)
            dxe_Saha_dlna = -(recomb_functions.rydberg / TCMB - 3./2.) * xe_Saha**2 / (2. * xe_Saha + s)

            # Compute xe using two-photon processes
            grad_dxedlna_func = grad(self.dxe_dlna_twophoton, argnums=0)
            grad_dxedlna = grad_dxedlna_func(xe_Saha, TCMB, TCMB, H, nH, 0.0)
            xe = xe_Saha + dxe_Saha_dlna / grad_dxedlna

            # Store current xe value in the output array
            xe_output = xe_output.at[iz].set(xe)
            lna_output = lna_output.at[iz].set(lna)

            # Check difference
            diff = jnp.abs(xe_Saha - xe)
            stop = diff > threshold  # Stop when diff < threshold

            # Increment index
            iz = iz + 1

            return (xe_output, lna_output, xe, iz, stop)

        def stop_condition(state):
            _, _, _, iz, stop = state
            return (iz < self.concrete_axis_size.size) & (~stop)  # Continue until stop condition is met or we run out of space

        # Initial state: (xe_output, xe, iz, stop flag)
        initial_state = (xe_output, lna_output, xe, iz, stop)

        # Run the while loop until the stop condition is met
        final_state = lax.while_loop(stop_condition, compute_xe, initial_state)

        # Unpack the final state
        xe_output_final, lna_output_final, _, _, _ = final_state

        # Return the electron fraction array and the stopping `lna` value
        return xe_output_final, lna_output_final
    
    def xe_derivative_twophoton(self, lna, xe, args):
        """
        Compute ionization fraction derivative including two-photon processes.

        Derivative function for hydrogen ionization fraction evolution
        including two-photon transitions and correction functions.

        Parameters:
        -----------
        lna : float
            Log scale factor
        xe : float
            Current ionization fraction
        args : tuple
            h, omega_b, omega_cdm, Neff, YHe; the Hubble parameter,
            the baryon denisty Omega_b h^2, the CDM density Omega_cdm h^2, 
            the effecgive number of neutrinos, and the helium fraction

        Returns:
        --------
        float
            Time derivative dxe/dlna (units: dimensionless)
        """
        h, omega_b, omega_cdm, Neff, YHe = args
        omega_rad = cosmology.omega_rad0(Neff)
    
        z = 1. / jnp.exp(lna) - 1.
        x1s = 1. - xe                # fraction of neutral hydrogen
        TCMB = cosmology.TCMB(z)     # eV
        nH = cosmology.nH(z, omega_b, YHe)  # hydrogen number density, 1/cm^3
        H = cosmology.Hubble(z, h, omega_b, omega_cdm, omega_rad)  # Hubble parameter, 1/s
        GammaC = recomb_functions.Gamma_compton(xe, TCMB, YHe)  # Compton scattering rate, 1/s

        Tm = TCMB * (1.-H/GammaC)

        Delta = self.get_current_correction_func(TCMB, omega_b, omega_cdm, YHe, Neff)
        dxedlna = self.dxe_dlna_twophoton(xe, TCMB, Tm, H, nH, Delta)

        return dxedlna


    def solve_emla_twophoton(self, lna_axis_init, lna_axis_final, xe0, h, omega_b, omega_cdm, Neff, YHe, rtol=1e-6, atol=1e-9,solver=Kvaerno3(),max_steps=4096):
        """
        Solve HYREC-2 EMLA evolution with two-photon processes.

        Integrates hydrogen recombination including effective multilevel atom
        approximation with two-photon transitions and correction functions.

        Parameters:
        -----------
        lna_axis_init : float
            Initial log scale factor
        lna_axis_final : float
            Final log scale factor
        xe0 : float
            Initial ionization fraction
        h : float
            Hubble parameter
        omega_b : float
            The baryon density Omega_b h^2
        omega_cdm : float
            The density of Cold Dark Matter Omega_cdm h^2
        Neff : float
            Effective number of neutrinos
        YHe : float
            Helium fraction
        rtol : float, optional
            Relative tolerance (default: 1e-6)
        atol : float, optional
            Absolute tolerance (default: 1e-9)
        solver : diffrax.Solver, optional
            ODE solver (default: Kvaerno3())
        max_steps : int, optional
            Maximum steps (default: 1024)

        Returns:
        --------
        tuple
            (xe_output, lna_output) - ionization fraction and log scale factor arrays
        """
        # Initial conditions
        TCMB_init = cosmology.TCMB(jnp.exp(-lna_axis_init ) - 1.)  # Initial CMB temperature
        initial_state = xe0
        term = ODETerm(self.xe_derivative_twophoton)

        t0 = lna_axis_init
        # t1 = lna_axis_final
        t1 = jnp.inf

        # don't want to double count the boundary lna, so start saving after one step
        t_arr = jnp.linspace(t0+self.integration_spacing, t0+2*max_steps*self.integration_spacing, 2*max_steps)

        save_at = SaveAt(ts=t_arr) 
        adjoint=ForwardMode()

        def lna_check(state, **kwargs):
            lna = state.tprev
            return lna > lna_axis_final
        
        # use diffrax default max_steps of 4096
        sol = diffeqsolve(
            term, solver, t0=t0, t1=t1, dt0=1e-3, 
            y0=initial_state, 
            args=(h, omega_b, omega_cdm, Neff, YHe),
            stepsize_controller=PIDController(rtol, atol),saveat=save_at,
            discrete_terminating_event = DiscreteTerminatingEvent(lna_check),
            adjoint=adjoint
        )
        
        xe_output = sol.ys
        lna_output = sol.ts

        return xe_output, lna_output


    def xe_tm_derivative(self, lna, state, args):
        """
        Compute coupled derivatives for ionization fraction and matter temperature.

        Derivative function for simultaneous evolution of hydrogen ionization
        fraction and matter temperature including Compton heating/cooling.

        Parameters:
        -----------
        lna : float
            Log scale factor
        state : array
            Current state [xe, Tm]
        args : tuple
            h, omega_b, omega_cdm, Neff, YHe; the Hubble parameter,
            the baryon denisty Omega_b h^2, the CDM density Omega_cdm h^2, 
            the effecgive number of neutrinos, and the helium fraction

        Returns:
        --------
        array
            Time derivatives [dxe/dlna, dTm/dlna] (units: dimensionless, eV)
        """
        xe, Tm = state
        h, omega_b, omega_cdm, Neff, YHe = args
        omega_rad = cosmology.omega_rad0(Neff)
        
        z = 1. / jnp.exp(lna) - 1.   # redshift z
        TCMB = cosmology.TCMB(z)      # eV
        nH = cosmology.nH(z, omega_b, YHe)  # hydrogen number density, 1/cm^3
        H = cosmology.Hubble(z, h, omega_b, omega_cdm, omega_rad)  # Hubble parameter, 1/s
        GammaC = recomb_functions.Gamma_compton(xe, TCMB, YHe)  # Compton scattering rate, 1/s

        Delta = 0.0
        dxedlna = self.dxe_dlna_twophoton(xe, TCMB, Tm, H, nH, Delta)
        dTmdlna = (-2 * H * Tm + GammaC * (TCMB - Tm)) / H

        return jnp.array([dxedlna, dTmdlna])

    def solve_emla(self, lna_axis, xe0, h, omega_b, omega_cdm, Neff, YHe,rtol=1e-7, atol=1e-9,solver=Tsit5(),max_steps=4096):
        """
        Solve late-time EMLA evolution without two-photon processes.

        Integrates hydrogen recombination using effective multilevel atom
        approximation for late times when two-photon processes are negligible.

        Parameters:
        -----------
        lna_axis : array
            Log scale factor grid
        xe0 : float
            Initial ionization fraction
        h : float
            Hubble parameter
        omega_b : float
            The baryon density Omega_b h^2
        omega_cdm : float
            The density of Cold Dark Matter Omega_cdm h^2
        Neff : float
            Effective number of neutrinos
        YHe : float
            Helium fraction
        rtol : float, optional
            Relative tolerance (default: 1e-7)
        atol : float, optional
            Absolute tolerance (default: 1e-9)
        solver : diffrax.Solver, optional
            ODE solver (default: Tsit5())
        max_steps : int, optional
            Maximum steps (default: 4096)

        Returns:
        --------
        tuple
            (xe_output, Tm_output, lna_output) - ionization fraction, matter temperature, 
            and lna arrays
        """

        omega_rad = cosmology.omega_rad0(Neff)

        t0 = lna_axis.min()-self.integration_spacing # need to back up a step since that's where we specified xe0
        t1 = jnp.inf 

        # need to go at least twice max_steps to make sure we catch the t1 we actually want
        t_arr = jnp.linspace(t0+self.integration_spacing, t0+2*max_steps*self.integration_spacing, 2*max_steps)

        save_at = SaveAt(ts=t_arr) 

        TCMB_init = cosmology.TCMB(jnp.exp(-t0) - 1.) 

        Tm0 = TCMB_init * (1.-cosmology.Hubble(1/jnp.exp(t0) - 1, h, omega_b, omega_cdm, omega_rad)/recomb_functions.Gamma_compton(xe0, TCMB_init, YHe))

        initial_state = jnp.array([xe0, Tm0])
        term = ODETerm(self.xe_tm_derivative)
        adjoint=ForwardMode()

        def temperature_check(t, y, args, **kwargs):
            lna = t
            _, Tm = y
            z = jnp.exp(-lna) - 1
            TCMB = cosmology.TCMB(z)
            TR_MIN = 0.004   # Minimum Tr in eV 
            T_RATIO_MIN = 0.1   # Minimum Tratio 
            ratio = jnp.minimum(Tm / TCMB, TCMB / Tm)
            return jnp.logical_or(TCMB < TR_MIN, ratio < T_RATIO_MIN) # stop when true
        
        event = Event(temperature_check)

        sol = diffeqsolve(
            term, solver, t0=t0, t1=t1, dt0=1e-3, 
            y0=initial_state, 
            args=(h, omega_b, omega_cdm, Neff, YHe),
            stepsize_controller=PIDController(rtol, atol),saveat=save_at,
            adjoint=adjoint,
            max_steps=max_steps,
            event = event
        )


        xe_output = jnp.where(jnp.isnan(sol.ys[:, 0]) , jnp.inf, sol.ys[:, 0])
        Tm_output = jnp.where(jnp.isnan(sol.ys[:, 1]) , jnp.inf, sol.ys[:, 1])

        return xe_output, Tm_output, jnp.where(jnp.isnan(sol.ts), jnp.inf, sol.ts)


    def dxe_dlna_twophoton(self, xe, TCMB, Tm, H, nH, Delta):

        """
        Compute two-photon recombination rate.

        Calculates ionization fraction evolution rate including two-photon
        transitions using effective multilevel atom approximation.

        Parameters:
        -----------
        xe : float
            Current ionization fraction
        TCMB : float
            CMB temperature (units: eV)
        Tm : float
            Matter temperature (units: eV)
        H : float
            Hubble parameter (units: s^{-1})
        nH : float
            Hydrogen number density (units: cm^{-3})
        Delta : float
            Correction function value

        Returns:
        --------
        float
            Recombination rate dxe/dlna (units: dimensionless)
        """

        x1s = 1.-xe

        # Interpolate transition rates
        A2s, A2p, B2s, B2p, _, _, R2p2s, R2s2p = recomb_functions.effective_coefficients(TCMB, Tm, H, nH, x1s)    

        # Compute the matrix and source vector forms of steady state equations, then solve the linear system for real and virtual populations.
        T, S = self.steady_state_equations(xe, H, nH, TCMB, A2s, A2p, B2s, B2p, R2p2s, R2s2p, Delta)
        X    = jnp.linalg.solve(T, S)
        x2s  = X[0]
        x2p  = X[1]

        return (x2s*B2s + x2p*B2p - xe**2*nH*(A2s+A2p)) / H

    def get_current_correction_func(self, TCMB, omega_b, omega_cdm, YHe, Neff):
        """
        Interpolate correction function for current cosmology.

        Interpolates correction function and applies cosmological parameter
        derivatives at current CMB temperature for accurate recombination rates.

        Parameters:
        -----------
        TCMB : float
            CMB temperature (units: eV)
        h : float
            Hubble parameter
        omega_b : float
            The baryon density Omega_b h^2
        omega_cdm : float
            The density of Cold Dark Matter Omega_cdm h^2
        Neff : float
            Effective number of neutrinos
        YHe : float
            Helium fraction

        Returns:
        --------
        float
            Correction function value (units: dimensionless)
        """
        # Fiducial cosmology values at which the correction functions were tabulated.
        omega_H_fid  = 0.01689
        omega_cb_fid = 0.14175
        Neff_fid     = 3.046


        # For the user inputed cosmology currently scanned over.
        omega_H  = omega_b*(1-YHe)
        omega_cb = omega_b + omega_cdm

        Delta        = jnp.interp(TCMB, kB*self.swift[:, 0], self.swift[:, 1])
        dDelta_domcb = jnp.interp(TCMB, kB*self.swift[:, 0], self.swift[:, 2])
        dDelta_domH  = jnp.interp(TCMB, kB*self.swift[:, 0], self.swift[:, 3])
        dDelta_dNeff = jnp.interp(TCMB, kB*self.swift[:, 0], self.swift[:, 4])
        
        return Delta + (omega_cb-omega_cb_fid)*dDelta_domcb + (omega_H-omega_H_fid)*dDelta_domH + (Neff-Neff_fid)*dDelta_dNeff

    def steady_state_equations(self, xe, H, nH, TCMB, A2s, A2p, B2s, B2p, R2p2s, R2s2p, Delta):
        """
        Set up steady-state level population equations.

        Constructs matrix equation for hydrogen level populations in
        steady-state approximation for 2s and 2p levels.

        Parameters:
        -----------
        xe : float
            Current ionization fraction
        H : float
            Hubble parameter (units: s^{-1})
        nH : float
            Hydrogen number density (units: cm^{-3})
        TCMB : float
            CMB temperature (units: eV)
        A2s : float
            2s recombination coefficient (units: cm^3 s^{-1})
        A2p : float
            2p recombination coefficient (units: cm^3 s^{-1})
        B2s : float
            2s photoionization coefficient (units: s^{-1})
        B2p : float
            2p photoionization coefficient (units: s^{-1})
        R2p2s : float
            2p→2s transition rate (units: s^{-1})
        R2s2p : float
            2s→2p transition rate (units: s^{-1})
        Delta : float
            Correction function value

        Returns:
        --------
        tuple
            (T, S) - transition matrix and source vector for level populations
        """
        T = jnp.zeros((2, 2), dtype="float64") 
        S = jnp.zeros(2, dtype="float64")

        x1s   = 1.-xe      # Recombined hydrogen fraction.
        
        # List of transition rates needed
        RLya  = self.Lyn_esc_rate(2, H, nH, x1s)  # Lyman-alpha escape rate.     
        R2s1s = 8.2206     # Two-photon transition rate from 2s to 1s.
        R2p1s = RLya / (1.+Delta) # Two-photon transition rate from 2p to 1s, with HYREC-2 fitting correction function.
        R1s2s = jnp.exp(-recomb_functions.E21/TCMB)*R2s1s
        R1s2p = 3.*jnp.exp(-recomb_functions.E21/TCMB)*R2p1s

        # Upper 2x2 part of T matrix.
        T = T.at[0, 0].set(B2s+R2s2p+R2s1s)
        T = T.at[0, 1].set(-R2p2s)
        T = T.at[1, 0].set(-R2s2p)
        T = T.at[1, 1].set(B2p+R2p2s+R2p1s)

        # First 2 entries of source vector elements
        S = S.at[0].set(xe**2*nH*A2s+x1s*R1s2s)
        S = S.at[1].set(xe**2*nH*A2p+x1s*R1s2p)

        return (T, S)

    def Lyn_esc_rate(self, n, H, nH, x1s):
        """
        Computes the Lyman-n escape rate, rate at which photons redshift past the Lyman-n line 
        without being absorbed. We use the convention that n=2 is Ly-alpha, n=3 is Ly-beta...

        Parameters
        ----------
        n : float
            Requested Lyman transition level, should be greater than 2.
        H : float
            Hubble parameter in s^-1.
        nH : float
            Hydrogen number density in cm^-3.
        x1s : float
            Fraction of 1s bound hydrogen.

        Returns
        -------
        RLyn : float
            Rate of escape of Lyman-n level.
        """
        lambda_lya = 2.*jnp.pi*hbar*c / recomb_functions.rydberg * 4./3. # Lyman-alpha Wavelength
        RLya = 8.*jnp.pi*H/3./nH/x1s/lambda_lya**3      # Rate of escape of Lyman-alpha
        RLyn = (4*(n**2-1)/3/n**2)**3 * RLya            # (lambda_lya/lambda_lyn)^3 * RLya
        return RLyn
    

  
    def TLA_xe_deriv(self, lna, state, args):
        """
        Compute coupled derivatives for ionization fraction and matter temperature
        using Peebles three-level atom.

        Parameters:
        -----------
        lna : float
            Log scale factor
        state : array
            Current state [xe, Tm]
        args : tuple
            h, omega_b, omega_cdm, Neff, YHe; the Hubble parameter,
            the baryon denisty Omega_b h^2, the CDM density Omega_cdm h^2, 
            the effecgive number of neutrinos, and the helium fraction

        Returns:
        --------
        array
            Time derivatives [dxe/dlna, dTm/dlna] (units: dimensionless, eV)
        """
        xe, Tm  = state
        h, omega_b, omega_cdm, Neff, YHe = args

        xHII = xe # since everything else is fully recombined
        z = jnp.exp(-lna) - 1
        omega_rad = cosmology.omega_rad0(Neff)
        nH = cosmology.nH(z, omega_b, YHe)
        H = cosmology.Hubble(z, h, omega_b, omega_cdm, omega_rad)
        TCMB = cosmology.TCMB(z) 

        C = recomb_functions.peebles_C(z, xHII, H, nH)
        alpha = recomb_functions.alpha_H(Tm)                     
        beta  = recomb_functions.beta_H(Tm)                  

        # dxe/d(lna) = (1/H) * dxe/dt
        dxe_dt = C * (beta * (1.0 - xe) - alpha * nH * xe**2)
        dxe_dloga = dxe_dt / H

        dTm_dloga = -2.0 * Tm + (recomb_functions.Gamma_compton(xe, TCMB, YHe) / H) * (TCMB - Tm)

        return jnp.array([dxe_dloga, dTm_dloga])
    
    def solve_TLA(self, lna0, lna_axis, xe0, Tm0, h, omega_b, omega_cdm, Neff, YHe, rtol=1e-7, atol=1e-9, solver=Kvaerno3(), max_steps = 4096):
        """
        Solve late-time TLA evolution.

        Integrates hydrogen recombination using Peebles TLA in the region
        beyond where SWIFT corrections are tabulated.

        Parameters:
        -----------
        lna0 : float
            Starting log scale factor
        lna_axis : array
            Log scale factor grid
        xe0 : float
            Initial ionization fraction
        Tm0: float
            Starting matter temperature
        h : float
            Hubble parameter
        omega_b : float
            The baryon density Omega_b h^2
        omega_cdm : float
            The density of Cold Dark Matter Omega_cdm h^2
        Neff : float
            Effective number of neutrinos
        YHe : float
            Helium fraction
        rtol : float, optional
            Relative tolerance (default: 1e-7)
        atol : float, optional
            Absolute tolerance (default: 1e-9)
        solver : diffrax.Solver, optional
            ODE solver (default: Tsit5())
        max_steps : int, optional
            Maximum steps (default: 4096)

        Returns:
        --------
        tuple
            (xe_output, Tm_output, lna_output) - ionization fraction, matter temperature, 
            and log scale factor arrays
        """
        t0 = lna0
        t1 = jnp.inf # lna_axis.max

        # need to go at least twice max_steps to make sure we catch t1
        t_arr = jnp.linspace(t0+self.integration_spacing, t0+2*max_steps*self.integration_spacing, 2*max_steps)

        save_at = SaveAt(ts=t_arr) 
        # save_at = SaveAt(ts=lna_axis) # but start saving output at step 1 or later

        initial_state = jnp.array([xe0, Tm0])
        term = ODETerm(self.TLA_xe_deriv)
        adjoint=ForwardMode()

        def lna_check(t, y, args, **kwargs):
            return t > jnp.max(lna_axis) # stop when true
        
        event = Event(lna_check)

        sol = diffeqsolve(
            term, solver, t0=t0, t1=t1, dt0=1e-3, 
            y0=initial_state, 
            args=(h, omega_b, omega_cdm, Neff, YHe),
            stepsize_controller=PIDController(rtol, atol),saveat=save_at,
            adjoint=adjoint,
            max_steps=max_steps,
            event=event
        )
        
        xe_output = jnp.where(jnp.isnan(sol.ys[:, 0]) , jnp.inf, sol.ys[:, 0])
        Tm_output = jnp.where(jnp.isnan(sol.ys[:, 1]) , jnp.inf, sol.ys[:, 1])

        return xe_output, Tm_output, jnp.where(jnp.isnan(sol.ts), jnp.inf, sol.ts)