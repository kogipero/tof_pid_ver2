import awkward as ak
import helper_functions as myfunc
import ROOT as r
import numpy as np

class TOFPlotter:
    def __init__(self, rootfile, name: str):
        self.rootfile = rootfile
        self.name = name


    def plot_tof_info(self, tof_pos_x, tof_pos_y, tof_pos_z, tof_time, tof_phi, tof_theta, tof_r, area=''):
        print('Start plotting TOF info')

        myfunc.make_histogram_root(ak.flatten(tof_pos_x),
                        100,
                        hist_range=[-1000, 1000],
                        title=f'TOF_rec_hit_pos_x_{area}',
                        xlabel='x [mm]',
                        ylabel='Entries',
                        outputname=f'{self.name}/tof_x',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(tof_pos_y),
                        100,
                        hist_range=[-1000, 1000],
                        title=f'TOF_rec_hit_pos_y_{area}',
                        xlabel='y [mm]',
                        ylabel='Entries',
                        outputname=f'{self.name}/tof_y',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(tof_pos_z),
                        100,
                        hist_range=[-2000, 2000],
                        title=f'TOF_rec_hit_pos_z_{area}',
                        xlabel='z [mm]',
                        ylabel='Entries',
                        outputname=f'{self.name}/tof_z',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(tof_time),
                        100,
                        hist_range=[0, 100],
                        title=f'TOF_rec_hit_time_{area}',
                        xlabel='time [ns]',
                        ylabel='Entries',
                        outputname=f'{self.name}/tof_time',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(tof_phi),
                        100,
                        hist_range=[-3.2, 3.2],
                        title=f'TOF_rec_hit_pos_phi_{area}',
                        xlabel='phi [rad]',
                        ylabel='Entries',
                        outputname=f'{self.name}/tof_phi',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(tof_theta),
                        100,
                        hist_range=[0, 3.2],
                        title=f'TOF_rec_hit_pos_theta_{area}',
                        xlabel='theta [rad]',
                        ylabel='Entries',
                        outputname=f'{self.name}/tof_theta',
                        rootfile=self.rootfile
                        )

        myfunc.make_histogram_root(ak.flatten(tof_r),
                        100,
                        hist_range=[0, 1000],
                        title=f'TOF_rec_hit_pos_r_{area}',
                        xlabel='r [mm]',
                        ylabel='Entries',
                        outputname=f'{self.name}/tof_r',
                        rootfile=self.rootfile
                        )

        print('End plotting TOF info')