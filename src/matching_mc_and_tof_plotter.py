import numpy as np
import ROOT as r
import helper_functions as myfunc

class MatchingMCAndTOFPlotter:
    def __init__(self, rootfile, name: str):
        self.rootfile = rootfile
        self.name = name

    def plot_matching_mc_and_tof(self, btof_hit_info_df):
        """""
        """""
        print('Start plotting matching MC and TOF info')

        myfunc.make_histogram_root(
            btof_hit_info_df['mc_pdg'],
            500, 
            hist_range=[-250, 500],
            title='MC_PDG_ID_matching_mc_and_TOF',
            xlabel='PDG_ID',
            ylabel='Entries',
            outputname=f'{self.name}/mc_pdg',
            rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            btof_hit_info_df['mc_momentum'],
            50, 
            hist_range=[0, 5],
            title='MC_momentum_matching_mc_and_TOF',
            xlabel='p [GeV]',
            ylabel='Entries',
            outputname=f'{self.name}/mc_momentum',
            rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            btof_hit_info_df['mc_vertex_x'],
            100,
            hist_range=[-200, 200],
            title='MC_vertex_x_matching_mc_and_TOF',
            xlabel='x [cm]',
            ylabel='Entries',
            outputname=f'{self.name}/mc_vertex_x',
            rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            btof_hit_info_df['mc_vertex_y'],
            100,
            hist_range=[-200, 200],
            title='MC_vertex_y_matching_mc_and_TOF',
            xlabel='y [cm]',
            ylabel='Entries',
            outputname=f'{self.name}/mc_vertex_y',
            rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            btof_hit_info_df['mc_vertex_z'],
            100,
            hist_range=[-200, 200],
            title='MC_vertex_z_matching_mc_and_TOF',
            xlabel='z [cm]',
            ylabel='Entries',
            outputname=f'{self.name}/mc_vertex_z',
            rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            btof_hit_info_df['btof_time'],
            100,
            hist_range=[0, 100],
            title='TOF_raw_hit_time_matching_mc_and_TOF',
            xlabel='time [ns]',
            ylabel='Entries',
            outputname=f'{self.name}/tof_time',
            rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            btof_hit_info_df['btof_pos_phi'],
            100,
            hist_range=[-3.2, 3.2],
            title='TOF_raw_hit_pos_phi_matching_mc_and_TOF',
            xlabel='phi [rad]',
            ylabel='Entries',
            outputname=f'{self.name}/tof_phi',
            rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            btof_hit_info_df['btof_pos_theta'],
            100,
            hist_range=[0, 3.2],
            title='TOF_raw_hit_pos_theta_matching_mc_and_TOF',
            xlabel='theta [rad]',
            ylabel='Entries',
            outputname=f'{self.name}/tof_theta',
            rootfile=self.rootfile
        )

    def plot_filtered_stable_particle_hit_and_generated_point(self, filtered_stable_btof_hit_info):
        """""
        """""
        print('Start plotting filtered stable particle hit and generated point')

        myfunc.make_histogram_root(
            filtered_stable_btof_hit_info['mc_pdg'],
            500,
            hist_range=[-250, 500],
            title='MC_PDG_ID_filtered_stable_particle_hit_and_generated_point',
            xlabel='PDG_ID',
            ylabel='Entries',
            outputname=f'{self.name}/mc_pdg_filtered_stable_particle_hit_and_generated_point',
            rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            filtered_stable_btof_hit_info['mc_momentum'],
            50,
            hist_range=[0, 5],
            title='MC_momentum_filtered_stable_particle_hit_and_generated_point',
            xlabel='p [GeV]',
            ylabel='Entries',
            outputname=f'{self.name}/mc_momentum_filtered_stable_particle_hit_and_generated_point',
            rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            filtered_stable_btof_hit_info['mc_vertex_x'],
            100,
            hist_range=[-200, 200],
            title='MC_vertex_x_filtered_stable_particle_hit_and_generated_point',
            xlabel='x [cm]',
            ylabel='Entries',
            outputname=f'{self.name}/mc_vertex_x_filtered_stable_particle_hit_and_generated_point',
            rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            filtered_stable_btof_hit_info['mc_vertex_y'],
            100,
            hist_range=[-200, 200],
            title='MC_vertex_y_filtered_stable_particle_hit_and_generated_point',
            xlabel='y [cm]',
            ylabel='Entries',
            outputname=f'{self.name}/mc_vertex_y_filtered_stable_particle_hit_and_generated_point',
            rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            filtered_stable_btof_hit_info['mc_vertex_z'],
            100,
            hist_range=[-200, 200],
            title='MC_vertex_z_filtered_stable_particle_hit_and_generated_point',
            xlabel='z [cm]',
            ylabel='Entries',
            outputname=f'{self.name}/mc_vertex_z_filtered_stable_particle_hit_and_generated_point',
            rootfile=self.rootfile
        )

        myfunc.make_histogram_root(
            filtered_stable_btof_hit_info['time'],
            100,
            hist_range=[0, 100],
            title='TOF_raw_hit_time_filtered_stable_particle_hit_and_generated_point',
            xlabel='time [ns]',
            ylabel='Entries',
            outputname=f'{self.name}/tof_time_filtered_stable_particle_hit_and_generated_point',
            rootfile=self.rootfile
        )