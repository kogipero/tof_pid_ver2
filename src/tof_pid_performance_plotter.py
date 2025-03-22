import numpy as np
import ROOT as r
import helper_functions as myfunc


class TOFPIDPerformancePlotter:
    def __init__(self, rootfile, name: str):
        self.rootfile = rootfile
        self.name = name

    def plot_tof_pid_performance(self, track_momentums_on_btof, track_momentum_on_ectof, btof_beta_inversees, btof_calc_mass):
        """
        Plots TOF PID performance.
        """
        print('Start plotting TOF PID performance')

        myfunc.make_histogram_root(
            track_momentums_on_btof,
                           100,
                           hist_range=[0, 5],
                        title='BTOF_Momentum_PID_Performance',
                        xlabel='Momentum [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_momentum_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            track_momentum_on_ectof,
                           100,
                           hist_range=[0, 5],
                        title='ETOF_Momentum_PID_Performance',
                        xlabel='Momentum [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/etof_momentum_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            btof_beta_inversees,
                        100,
                        hist_range=[0.8, 1.8],
                        title='BTOF_Beta_Inverse_PID_Performance',
                        xlabel='Beta Inverse',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_beta_inverse_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            btof_calc_mass,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF_Calculated_Mass',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_mass_pid_performance',
                        rootfile=self.rootfile
        )

        print('End plotting TOF PID performance')

    def plot_tof_pid_reconstruction_mass(self, 
                                         pi_calc_mass_on_btof, 
                                         k_calc_mass_on_btof, 
                                         p_calc_mass_on_btof, 
                                         e_calc_mass_on_btof, 
                                         track_momentums_on_btof,
                                         btof_beta_inversees,
                                         track_momentums_pi_on_btof, 
                                         track_momentums_k_on_btof, 
                                         track_momentums_p_on_btof, 
                                         track_momentums_e_on_btof, 
                                         btof_pi_beta_inversees, 
                                         btof_k_beta_inversees, 
                                         btof_p_beta_inversees, 
                                         btof_e_beta_inversees
                                         ):
        """
        Plots TOF PID mass reconstruction.
        """
        print('Start plotting TOF PID mass reconstruction')

        myfunc.make_histogram_root(
            pi_calc_mass_on_btof,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF_Calculated_Mass_for_Pi',
                        xlabel='Mass [GeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_mass_pi_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            k_calc_mass_on_btof,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF_Calculated_Mass_for_K',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_mass_k_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            p_calc_mass_on_btof,
                        100,
                        hist_range=[200, 1200],
                        title='BTOF_Calculated_Mass_for_P',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_mass_p_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            e_calc_mass_on_btof,
                        100,
                        hist_range=[0, 1000],
                        title='BTOF_Calculated_Mass_for_e',
                        xlabel='Mass [MeV]',
                        ylabel='Entries',
                        outputname=f'{self.name}/btof_mass_e_pid_performance',
                        rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_on_btof,
            100,
            [0, 3.5],
            btof_beta_inversees,
            100,
            [0.8, 3.5],
            title='BTOF_Momentum_vs_Beta_Inverse',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{self.name}/btof_momentum_vs_beta_inverse_pid_performance',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_on_btof,
            100,
            [0, 5],
            btof_beta_inversees,
            100,
            [0.8, 1.8],
            title='BTOF_Momentum_vs_Beta_Inverse',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{self.name}/btof_momentum_vs_beta_inverse_pid_performance_diff_range',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_pi_on_btof,
            100,
            [0, 3.5],
            btof_pi_beta_inversees,
            100,
            [0.8, 1.8],
            title='BTOF_Momentum_vs_Beta_Inverse_for_Pi',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{self.name}/btof_momentum_vs_beta_inverse_pi_pid_performance',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_k_on_btof,
            100,
            [0, 3.5],
            btof_k_beta_inversees,
            100,
            [0.8, 1.8],
            title='BTOF_Momentum_vs_Beta_Inverse_for_K',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{self.name}/btof_momentum_vs_beta_inverse_k_pid_performance',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_p_on_btof,
            100,
            [0, 3.5],
            btof_p_beta_inversees,
            100,
            [0.8, 1.8],
            title='BTOF_Momentum_vs_Beta_Inverse_for_P',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{self.name}/btof_momentum_vs_beta_inverse_p_pid_performance',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            track_momentums_e_on_btof,
            100,
            [0, 3.5],
            btof_e_beta_inversees,
            100,
            [0.8, 1.8],
            title='BTOF_Momentum_vs_Beta_Inverse_for_e',
            xlabel='Momentum [GeV]',
            ylabel='Beta Inverse',
            outputname=f'{self.name}/btof_momentum_vs_beta_inverse_e_pid_performance',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        print('End plotting TOF PID mass reconstruction')

    def miss_id_particle_plot(self,
                              correct_id_pi,
                              correct_id_k,
                              miss_id_pi_as_k,
                              miss_id_k_as_pi,
    ):
        """
        Plot the miss identification of particles.
        """
        print('Start plotting miss identification of particles')

        correct_pi_momentum = correct_id_pi["momentum"]
        correct_k_momentum = correct_id_k["momentum"]
        miss_pi_as_k_momentum = miss_id_pi_as_k["momentum"]
        miss_k_as_pi_momentum = miss_id_k_as_pi["momentum"]

        correct_pi_time = correct_id_pi["time"]
        correct_k_time = correct_id_k["time"]
        miss_pi_as_k_time = miss_id_pi_as_k["time"]
        miss_k_as_pi_time = miss_id_k_as_pi["time"]

        correct_pi_mass = correct_id_pi["mass"]
        correct_k_mass = correct_id_k["mass"]
        miss_pi_as_k_mass = miss_id_pi_as_k["mass"]
        miss_k_as_pi_mass = miss_id_k_as_pi["mass"]

        correct_pi_mc_momentum = correct_id_pi["mc_momentum"]
        correct_k_mc_momentum = correct_id_k["mc_momentum"]
        miss_pi_as_k_mc_momentum = miss_id_pi_as_k["mc_momentum"]
        miss_k_as_pi_mc_momentum = miss_id_k_as_pi["mc_momentum"]

        correct_pi_mc_momentum_phi = correct_id_pi["mc_momentum_phi"]
        correct_k_mc_momentum_phi = correct_id_k["mc_momentum_phi"]
        miss_pi_as_k_mc_momentum_phi = miss_id_pi_as_k["mc_momentum_phi"]
        miss_k_as_pi_mc_momentum_phi = miss_id_k_as_pi["mc_momentum_phi"]

        correct_pi_mc_momentum_theta = correct_id_pi["mc_momentum_theta"]
        correct_k_mc_momentum_theta = correct_id_k["mc_momentum_theta"]
        miss_pi_as_k_mc_momentum_theta = miss_id_pi_as_k["mc_momentum_theta"]
        miss_k_as_pi_mc_momentum_theta = miss_id_k_as_pi["mc_momentum_theta"]

        correct_pi_mc_vertex_x = correct_id_pi["mc_vertex_x"]
        correct_k_mc_vertex_x = correct_id_k["mc_vertex_x"]
        miss_pi_as_k_mc_vertex_x = miss_id_pi_as_k["mc_vertex_x"]
        miss_k_as_pi_mc_vertex_x = miss_id_k_as_pi["mc_vertex_x"]

        correct_pi_mc_vertex_y = correct_id_pi["mc_vertex_y"]
        correct_k_mc_vertex_y = correct_id_k["mc_vertex_y"]
        miss_pi_as_k_mc_vertex_y = miss_id_pi_as_k["mc_vertex_y"]
        miss_k_as_pi_mc_vertex_y = miss_id_k_as_pi["mc_vertex_y"]

        correct_pi_mc_vertex_z = correct_id_pi["mc_vertex_z"]
        correct_k_mc_vertex_z = correct_id_k["mc_vertex_z"]
        miss_pi_as_k_mc_vertex_z = miss_id_pi_as_k["mc_vertex_z"]
        miss_k_as_pi_mc_vertex_z = miss_id_k_as_pi["mc_vertex_z"]

        correct_pi_mc_vertex_x = np.array(correct_pi_mc_vertex_x)
        correct_pi_mc_vertex_y = np.array(correct_pi_mc_vertex_y)
        correct_k_mc_vertex_x = np.array(correct_k_mc_vertex_x)
        correct_k_mc_vertex_y = np.array(correct_k_mc_vertex_y)
        miss_pi_as_k_mc_vertex_x = np.array(miss_pi_as_k_mc_vertex_x)
        miss_pi_as_k_mc_vertex_y = np.array(miss_pi_as_k_mc_vertex_y)
        miss_k_as_pi_mc_vertex_x = np.array(miss_k_as_pi_mc_vertex_x)
        miss_k_as_pi_mc_vertex_y = np.array(miss_k_as_pi_mc_vertex_y)


        correct_pi_mc_vertex_r = np.sqrt(1000* (correct_pi_mc_vertex_x**2 + correct_pi_mc_vertex_y**2))
        correct_k_mc_vertex_r = np.sqrt(1000* (correct_k_mc_vertex_x**2 + correct_k_mc_vertex_y**2))
        miss_pi_as_k_mc_vertex_r = np.sqrt(1000* (miss_pi_as_k_mc_vertex_x**2 + miss_pi_as_k_mc_vertex_y**2))
        miss_k_as_pi_mc_vertex_r = np.sqrt(1000* (miss_k_as_pi_mc_vertex_x**2 + miss_k_as_pi_mc_vertex_y**2))

        myfunc.make_stacked_histogram_root(
            [correct_pi_momentum, miss_pi_as_k_momentum],
            100,
            [0, 3.5],
            title='Miss_Identification_of_Particles_Momentum_pi_as_k',
            xlabel='Momentum [GeV]',
            ylabel='Entries',
            labels=['Correct Pi', 'Miss Pi as K'],
            outputname=f'{self.name}/miss_id_pi_as_k_momentum',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_k_momentum, miss_k_as_pi_momentum],
            100,
            [0, 3.5],
            title='Miss_Identification_of_Particles_Momentum_k_as_pi',
            xlabel='Momentum [GeV]',
            ylabel='Entries',
            labels=['Correct K', 'Miss K as Pi'],
            outputname=f'{self.name}/miss_id_k_as_pi_momentum',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_pi_time, miss_pi_as_k_time],
            100,
            [0, 20],
            title='Miss_Identification_of_Particles_Time_pi_as_k',
            xlabel='Time [ns]',
            ylabel='Entries',
            labels=['Correct Pi', 'Miss Pi as K'],
            outputname=f'{self.name}/miss_id_pi_as_k_time',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_k_time, miss_k_as_pi_time],
            100,
            [0, 20],
            title='Miss_Identification_of_Particles_Time_k_as_pi',
            xlabel='Time [ns]',
            ylabel='Entries',
            labels=['Correct K', 'Miss K as Pi'],
            outputname=f'{self.name}/miss_id_k_as_pi_time',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_pi_mass, miss_pi_as_k_mass],
            100,
            [0, 1000],
            title='Miss_Identification_of_Particles_Mass_pi_as_k',
            xlabel='Mass [MeV]',
            ylabel='Entries',
            labels=['Correct Pi', 'Miss Pi as K'],
            outputname=f'{self.name}/miss_id_pi_as_k_mass',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_k_mass, miss_k_as_pi_mass],
            100,
            [0, 1000],
            title='Miss_Identification_of_Particles_Mass_k_as_pi',
            xlabel='Mass [MeV]',
            ylabel='Entries',
            labels=['Correct K', 'Miss K as Pi'],
            outputname=f'{self.name}/miss_id_k_as_pi_mass',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_pi_mc_vertex_z, miss_pi_as_k_mc_vertex_z],
            100,
            [-50, 50],
            title='Miss_Identification_of_Particles_Vertex_Z_pi_as_k',
            xlabel='Vertex Z [cm]',
            ylabel='Entries',
            labels=['Correct Pi', 'Miss Pi as K'],
            outputname=f'{self.name}/miss_id_pi_as_k_vertex_z',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_k_mc_vertex_z, miss_k_as_pi_mc_vertex_z],
            100,
            [-50, 50],
            title='Miss_Identification_of_Particles_Vertex_Z_k_as_pi',
            xlabel='Vertex Z [cm]',
            ylabel='Entries',
            labels=['Correct K', 'Miss K as Pi'],
            outputname=f'{self.name}/miss_id_k_as_pi_vertex_z',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_pi_mc_momentum, miss_pi_as_k_mc_momentum],
            100,
            [0, 3.5],
            title='Miss_Identification_of_Particles_MC_Momentum_pi_as_k',
            xlabel='MC Momentum [GeV]',
            ylabel='Entries',
            labels=['Correct Pi', 'Miss Pi as K'],
            outputname=f'{self.name}/miss_id_pi_as_k_mc_momentum',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_k_mc_momentum, miss_k_as_pi_mc_momentum],
            100,
            [0, 3.5],
            title='Miss_Identification_of_Particles_MC_Momentum_k_as_pi',
            xlabel='MC Momentum [GeV]',
            ylabel='Entries',
            labels=['Correct K', 'Miss K as Pi'],
            outputname=f'{self.name}/miss_id_k_as_pi_mc_momentum',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_pi_mc_momentum_phi, miss_pi_as_k_mc_momentum_phi],
            100,
            [-3.5, 3.5],
            title='Miss_Identification_of_Particles_MC_Momentum_Phi_pi_as_k',
            xlabel='MC Momentum Phi [rad]',
            ylabel='Entries',
            labels=['Correct Pi', 'Miss Pi as K'],
            outputname=f'{self.name}/miss_id_pi_as_k_mc_momentum_phi',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_k_mc_momentum_phi, miss_k_as_pi_mc_momentum_phi],
            100,
            [-3.5, 3.5],
            title='Miss_Identification_of_Particles_MC_Momentum_Phi_k_as_pi',
            xlabel='MC Momentum Phi [rad]',
            ylabel='Entries',
            labels=['Correct K', 'Miss K as Pi'],
            outputname=f'{self.name}/miss_id_k_as_pi_mc_momentum_phi',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_pi_mc_momentum_theta, miss_pi_as_k_mc_momentum_theta],
            100,
            [0, 3.5],
            title='Miss_Identification_of_Particles_MC_Momentum_Theta_pi_as_k',
            xlabel='MC Momentum Theta [rad]',
            ylabel='Entries',
            labels=['Correct Pi', 'Miss Pi as K'],
            outputname=f'{self.name}/miss_id_pi_as_k_mc_momentum_theta',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_k_mc_momentum_theta, miss_k_as_pi_mc_momentum_theta],
            100,
            [0, 3.5],
            title='Miss_Identification_of_Particles_MC_Momentum_Theta_k_as_pi',
            xlabel='MC Momentum Theta [rad]',
            ylabel='Entries',
            labels=['Correct K', 'Miss K as Pi'],
            outputname=f'{self.name}/miss_id_k_as_pi_mc_momentum_theta',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_pi_mc_vertex_r, miss_pi_as_k_mc_vertex_r],
            100,
            [0, 20],
            title='Miss_Identification_of_Particles_MC_Vertex_R_pi_as_k',
            xlabel='MC Vertex R [cm]',
            ylabel='Entries',
            labels=['Correct Pi', 'Miss Pi as K'],
            outputname=f'{self.name}/miss_id_pi_as_k_mc_vertex_r',
            rootfile=self.rootfile
        )

        myfunc.make_stacked_histogram_root(
            [correct_k_mc_vertex_r, miss_k_as_pi_mc_vertex_r],
            100,
            [0, 20],
            title='Miss_Identification_of_Particles_MC_Vertex_R_k_as_pi',
            xlabel='MC Vertex R [cm]',
            ylabel='Entries',
            labels=['Correct K', 'Miss K as Pi'],
            outputname=f'{self.name}/miss_id_k_as_pi_mc_vertex_r',
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            correct_pi_mc_vertex_z,
            100,
            [-50, 50],
            correct_pi_mc_vertex_r,
            100,
            [0, 20],
            title='Correct_Pi_MC_Vertex_Z_vs_R',
            xlabel='MC Vertex Z [cm]',
            ylabel='MC Vertex R [cm]',
            outputname=f'{self.name}/correct_pi_mc_vertex_z_vs_r',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            correct_k_mc_vertex_z,
            100,
            [-50, 50],
            correct_k_mc_vertex_r,
            100,
            [0, 20],
            title='Correct_K_MC_Vertex_Z_vs_R',
            xlabel='MC Vertex Z [cm]',
            ylabel='MC Vertex R [cm]',
            outputname=f'{self.name}/correct_k_mc_vertex_z_vs_r',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            miss_pi_as_k_mc_vertex_z,
            100,
            [-50, 50],
            miss_pi_as_k_mc_vertex_r,
            100,
            [0, 20],
            title='Miss_Pi_as_K_MC_Vertex_Z_vs_R',
            xlabel='MC Vertex Z [cm]',
            ylabel='MC Vertex R [cm]',
            outputname=f'{self.name}/miss_pi_as_k_mc_vertex_z_vs_r',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        myfunc.make_2Dhistogram_root(
            miss_k_as_pi_mc_vertex_z,
            100,
            [-50, 50],
            miss_k_as_pi_mc_vertex_r,
            100,
            [0, 20],
            title='Miss_K_as_Pi_MC_Vertex_Z_vs_R',
            xlabel='MC Vertex Z [cm]',
            ylabel='MC Vertex R [cm]',
            outputname=f'{self.name}/miss_k_as_pi_mc_vertex_z_vs_r',
            cmap='plasma',
            logscale=True,
            rootfile=self.rootfile
        )

        print('End plotting miss identification of particles')

    def plot_separation_power_vs_momentum(
        self,
        btof_calc_mass: np.ndarray,
        btof_pdg: np.ndarray,
        track_momentums_on_btof: np.ndarray,
        track_momentums_transverse_on_btof: np.ndarray,
        nbins: int = 35,
        momentum_range: tuple = (0, 3.5),
    ):
        """

        """

        pi_mask = (btof_pdg ==  211) | (btof_pdg == -211)
        k_mask  = (btof_pdg ==  321) | (btof_pdg == -321)
        p_mask  = (btof_pdg == 2212) | (btof_pdg == -2212)

        pi_mass_all = btof_calc_mass[pi_mask]
        pi_mom_all  = track_momentums_transverse_on_btof[pi_mask]
        k_mass_all  = btof_calc_mass[k_mask]
        k_mom_all   = track_momentums_on_btof[k_mask]
        p_mass_all  = btof_calc_mass[p_mask]
        p_mom_all   = track_momentums_on_btof[p_mask]

        p_bins      = np.linspace(momentum_range[0], momentum_range[1], nbins+1)
        bin_centers = 0.5 * (p_bins[:-1] + p_bins[1:])
        separation_list_pi_k = []
        separation_list_k_p = []

        for i in range(nbins):
            p_low  = p_bins[i]
            p_high = p_bins[i+1]

            pi_in_bin = pi_mass_all[(pi_mom_all >= p_low) & (pi_mom_all < p_high)]
            k_in_bin  = k_mass_all[(k_mom_all  >= p_low) & (k_mom_all  < p_high)]
            p_in_bin  = p_mass_all[(p_mom_all  >= p_low) & (p_mom_all  < p_high)]


            if len(pi_in_bin) < 5 or len(k_in_bin) < 5:
                separation_list_pi_k.append(None)
                continue

            if len(k_in_bin) < 5 or len(p_in_bin) < 5:
                separation_list_k_p.append(None)
                continue

            hist_pi_name = f"hist_pi_bin_sep{i}"
            hist_pi = r.TH1F(hist_pi_name, ";Mass [MeV];Entries", 100, 0, 1000)
            for val in pi_in_bin:
                hist_pi.Fill(val)

            hist_pi.SetTitle(f"Pi Mass in {p_low:.2f} - {p_high:.2f} GeV")

            bin_max   = hist_pi.GetMaximumBin()
            x_max     = hist_pi.GetBinCenter(bin_max)  # peak position
            ampl      = hist_pi.GetBinContent(bin_max) # amplitude
            rms       = hist_pi.GetRMS()               # RMS

            f_pi = r.TF1("f_pi","[0]*exp(-0.5*((x-[1])/[2])**2)", 0, 1000)
            f_pi.SetParameters(ampl, x_max, rms)
            f_pi.SetParLimits(2, 1e-3, 200)  # limit sigma to be positive

            hist_pi.Fit(f_pi, "Q")
            A_pi    = f_pi.GetParameter(0)
            mu_pi   = f_pi.GetParameter(1)
            sigma_pi= f_pi.GetParameter(2)
            hist_k_name = f"hist_k_bin_sep{i}"
            hist_k = r.TH1F(hist_k_name, ";Mass [MeV];Entries", 100, 0, 1000)
            for val in k_in_bin:
                hist_k.Fill(val)

            hist_k.SetTitle(f"K Mass in {p_low:.2f} - {p_high:.2f} GeV")

            f_k = r.TF1("f_k","[0]*exp(-0.5*((x-[1])/[2])**2)", 0, 1000)
            f_k.SetParameters(hist_k.GetMaximum(), 490, 20) 
            hist_k.Fit(f_k, "Q")
            A_k     = f_k.GetParameter(0)
            mu_k    = f_k.GetParameter(1)
            sigma_k = f_k.GetParameter(2)

            hist_p_name = f"hist_p_bin_sep{i}"
            hist_p = r.TH1F(hist_p_name, ";Mass [MeV];Entries", 100, 0, 1000)
            for val in p_in_bin:
                hist_p.Fill(val)

            hist_p.SetTitle(f"P Mass in {p_low:.2f} - {p_high:.2f} GeV")

            f_p = r.TF1("f_p","[0]*exp(-0.5*((x-[1])/[2])**2)", 0, 1000)
            f_p.SetParameters(hist_p.GetMaximum(), 940, 20)
            hist_p.Fit(f_p, "Q")
            A_p     = f_p.GetParameter(0)
            mu_p    = f_p.GetParameter(1)
            sigma_p = f_p.GetParameter(2)

            # separation power
            sep_power_pi_k = None
            sep_power_k_p = None
            if sigma_pi>1e-7 and sigma_k>1e-7:
                sep_power_pi_k = abs(mu_pi - mu_k)/np.sqrt(1/2 * (sigma_pi**2 + sigma_k**2))

            if sigma_k>1e-7 and sigma_p>1e-7:
                sep_power_k_p = abs(mu_k - mu_p)/np.sqrt(1/2 * (sigma_k**2 + sigma_p**2))

            separation_list_pi_k.append(sep_power_pi_k)
            separation_list_k_p.append(sep_power_k_p)

            if self.rootfile:
                hist_pi.Write()
                hist_k.Write()
                hist_p.Write()
                # f_pi.Write()
                # f_k.Write()
                # f_p.Write()

        separation_list_pi_k_array = np.array(separation_list_pi_k, dtype=object)

        if len(separation_list_pi_k_array) < len(bin_centers):
            padding_size = len(bin_centers) - len(separation_list_pi_k_array)
            separation_list_pi_k_array = np.append(separation_list_pi_k_array, [None] * padding_size)

        valid_mask_pi_k = (separation_list_pi_k_array != None)

        if len(valid_mask_pi_k) != len(bin_centers):
            print(f"Warning: valid_mask_pi_k length {len(valid_mask_pi_k)} does not match bin_centers length {len(bin_centers)}")

        valid_sep_pi_k = separation_list_pi_k_array[valid_mask_pi_k].astype(float)
        valid_bin_center_pi_k = bin_centers[valid_mask_pi_k]

        gr1 = r.TGraph()
        gr1.SetName("sep_power_pi_k_vs_mom")
        gr1.SetTitle("Separation Power pi/k vs Momentum;pt [GeV];Separation Power")
        idx1 = 0
        for bc, sep in zip(valid_bin_center_pi_k, valid_sep_pi_k):
            gr1.SetPoint(idx1, bc, sep)
            idx1 += 1

        if self.rootfile:
            gr1.Write()  

        c1 = r.TCanvas("c1","Separation Power", 800,600)
        c1.SetLogy()

        gr1.GetXaxis().SetLimits(0, 3.5)
        gr1.GetYaxis().SetRangeUser(0, 50)
        gr1.Draw("AP")  
        gr1.SetMarkerStyle(20)
        gr1.SetMarkerColor(r.kRed)
        gr1.SetMarkerSize(1.3)

        sigma_line = r.TLine(0, 3, 3.5, 3)
        sigma_line.SetLineColor(r.kRed)
        sigma_line.SetLineStyle(2)
        sigma_line.Draw("same")

        c1.Update()

        #k/p

        separation_list_k_p_array = np.array(separation_list_k_p, dtype=object)

        if len(separation_list_k_p_array) < len(bin_centers):
            padding_size = len(bin_centers) - len(separation_list_k_p_array)
            separation_list_k_p_array = np.append(separation_list_k_p_array, [None] * padding_size)

        valid_mask_k_p = (separation_list_k_p_array != None)

        if len(valid_mask_k_p) != len(bin_centers):
            print(f"Warning: valid_mask_k_p length {len(valid_mask_k_p)} does not match bin_centers length {len(bin_centers)}")

        valid_sep_k_p = separation_list_k_p_array[valid_mask_k_p].astype(float)
        valid_bin_center_k_p = bin_centers[valid_mask_k_p]

        gr2 = r.TGraph()
        gr2.SetName("sep_power_k_p_vs_mom")
        gr2.SetTitle("Separation Power k/p vs Momentum;pt [GeV];Separation Power")
        idx2 = 0
        for bc, sep in zip(valid_bin_center_k_p, valid_sep_k_p):
            gr2.SetPoint(idx2, bc, sep)
            idx2 += 1

        if self.rootfile:
            gr2.Write()

        c2 = r.TCanvas("c2","Separation Power", 800,600)
        c2.SetLogy()

        gr2.GetXaxis().SetLimits(0, 3.5)
        gr2.GetYaxis().SetRangeUser(0, 50)
        gr2.Draw("AP")
        gr2.SetMarkerStyle(20)
        gr2.SetMarkerColor(r.kRed)
        gr2.SetMarkerSize(1.3)

        sigma_line = r.TLine(0, 3, 3.5, 3)
        sigma_line.SetLineColor(r.kRed)
        sigma_line.SetLineStyle(2)
        sigma_line.Draw("same")

        c2.Update()

        if self.rootfile:
            c1.Write("canvas_sep_power_logy")
            c2.Write("canvas_sep_power_logy")  
            
        if self.rootfile:
            gr = r.TGraph()
            gr.SetName("sep_power_vs_mom")
            idx = 0
            for bc, sep in zip(valid_bin_center_pi_k, valid_sep_pi_k):
                gr.SetPoint(idx, bc, sep)
                idx+=1
            gr.Write()

        return valid_bin_center_pi_k, valid_sep_pi_k
    

    def plot_purity_vs_momentum(
        self,
        bin_centers: np.ndarray,
        pi_eff_normal: np.ndarray,
        pi_eff_err_normal: np.ndarray,
        pi_eff_unique: np.ndarray,
        pi_eff_err_unique: np.ndarray,
        k_eff_normal: np.ndarray,
        k_eff_err_normal: np.ndarray,
        k_eff_unique: np.ndarray,
        k_eff_err_unique: np.ndarray,
        p_eff_normal: np.ndarray,
        p_eff_err_normal: np.ndarray,
        p_eff_unique: np.ndarray,
        p_eff_err_unique: np.ndarray,
        momentum_range: tuple = (0, 3.5),
    ):
        """
        Plot the purity of each particle as a function of momentum.
        """

        gr_pi_normal  = r.TGraphErrors()
        gr_pi_unique  = r.TGraphErrors()
        gr_pi_normal.SetName("pi_purity_normal")
        gr_pi_normal.SetTitle("Pi Purity (Normal);p [GeV];Purity")
        gr_pi_unique.SetName("pi_purity_unique")
        gr_pi_unique.SetTitle("Pi Purity (Unique);p [GeV];Purity")

        for ibin, (bc, eff_n, err_n, eff_u, err_u) in enumerate(zip(
            bin_centers, pi_eff_normal, pi_eff_err_normal, pi_eff_unique, pi_eff_err_unique
        )):
            gr_pi_normal.SetPoint(ibin, bc, eff_n)
            gr_pi_normal.SetPointError(ibin, 0, err_n)
            gr_pi_unique.SetPoint(ibin, bc, eff_u)
            gr_pi_unique.SetPointError(ibin, 0, err_u)

        gr_pi_normal.SetMarkerStyle(20)
        gr_pi_normal.SetMarkerColor(r.kRed)
        gr_pi_normal.SetLineColor(r.kRed)

        gr_pi_unique.SetMarkerStyle(21)
        gr_pi_unique.SetMarkerColor(r.kBlue)
        gr_pi_unique.SetLineColor(r.kBlue)

        c_pi = r.TCanvas("c_pi","Pi Purity",800,600)
        c_pi.Draw()
        frame_pi = c_pi.DrawFrame(0, 0, momentum_range[1], 1.05)
        frame_pi.GetXaxis().SetTitle("p [GeV]")
        frame_pi.GetYaxis().SetTitle("Purity")

        gr_pi_normal.Draw("P SAME")
        gr_pi_unique.Draw("P SAME")
        c_pi.BuildLegend()
        c_pi.Update()

        if self.rootfile:
            gr_pi_normal.Write()
            gr_pi_unique.Write()
            c_pi.Write("canvas_pi_purity")

        gr_k_normal  = r.TGraphErrors()
        gr_k_unique  = r.TGraphErrors()
        gr_k_normal.SetName("k_purity_normal")
        gr_k_normal.SetTitle("K Purity (Normal);p [GeV];Purity")
        gr_k_unique.SetName("k_purity_unique")
        gr_k_unique.SetTitle("K Purity (Unique);p [GeV];Purity")

        for ibin, (bc, eff_n, err_n, eff_u, err_u) in enumerate(zip(
            bin_centers, k_eff_normal, k_eff_err_normal, k_eff_unique, k_eff_err_unique
        )):
            gr_k_normal.SetPoint(ibin, bc, eff_n)
            gr_k_normal.SetPointError(ibin, 0, err_n)
            gr_k_unique.SetPoint(ibin, bc, eff_u)
            gr_k_unique.SetPointError(ibin, 0, err_u)

        gr_k_normal.SetMarkerStyle(20)
        gr_k_normal.SetMarkerColor(r.kGreen+2)
        gr_k_normal.SetLineColor(r.kGreen+2)

        gr_k_unique.SetMarkerStyle(21)
        gr_k_unique.SetMarkerColor(r.kOrange+1)
        gr_k_unique.SetLineColor(r.kOrange+1)

        c_k = r.TCanvas("c_k","K Purity",800,600)
        frame_k = c_k.DrawFrame(0,0,momentum_range[1],1.05)
        frame_k.GetXaxis().SetTitle("p [GeV]")
        frame_k.GetYaxis().SetTitle("Purity")
        gr_k_normal.Draw("P SAME")
        gr_k_unique.Draw("P SAME")
        c_k.BuildLegend()
        c_k.Update()

        if self.rootfile:
            gr_k_normal.Write()
            gr_k_unique.Write()
            c_k.Write("canvas_k_purity")

        gr_p_normal  = r.TGraphErrors()
        gr_p_unique  = r.TGraphErrors()
        gr_p_normal.SetName("p_purity_normal")
        gr_p_normal.SetTitle("Proton Purity (Normal);p [GeV];Purity")
        gr_p_unique.SetName("p_purity_unique")
        gr_p_unique.SetTitle("Proton Purity (Unique);p [GeV];Purity")

        for ibin, (bc, eff_n, err_n, eff_u, err_u) in enumerate(zip(
            bin_centers, p_eff_normal, p_eff_err_normal, p_eff_unique, p_eff_err_unique
        )):
            gr_p_normal.SetPoint(ibin, bc, eff_n)
            gr_p_normal.SetPointError(ibin, 0, err_n)
            gr_p_unique.SetPoint(ibin, bc, eff_u)
            gr_p_unique.SetPointError(ibin, 0, err_u)

        gr_p_normal.SetMarkerStyle(20)
        gr_p_normal.SetMarkerColor(r.kViolet)
        gr_p_normal.SetLineColor(r.kViolet)

        gr_p_unique.SetMarkerStyle(21)
        gr_p_unique.SetMarkerColor(r.kAzure+1)
        gr_p_unique.SetLineColor(r.kAzure+1)

        c_p = r.TCanvas("c_p","P Purity",800,600)
        frame_p = c_p.DrawFrame(0,0,momentum_range[1],1.05)
        frame_p.GetXaxis().SetTitle("p [GeV]")
        frame_p.GetYaxis().SetTitle("Purity")
        gr_p_normal.Draw("P SAME")
        gr_p_unique.Draw("P SAME")
        c_p.BuildLegend()
        c_p.Update()

        if self.rootfile:
            gr_p_normal.Write()
            gr_p_unique.Write()
            c_p.Write("canvas_p_purity")