import awkward as ak
import helper_functions as myfunc
import ROOT as r


class MatchingTOFAndTrackPlotter:
  def __init__(self, rootfile: str, name: str):
    """
    Constructor for the MatchingTOFAndTrackPlotter class."
    """
    self.name = name
    self.rootfile = rootfile

  def plot_matched_tof_and_track(
      self,
      btof_and_track_matched: dict,
      min_delta_angles_events: list,
      delta_angles_all: list,
  ):
    
    myfunc.make_histogram_root(
        btof_and_track_matched['track_p'],
        100,
        hist_range=[0, 5],
        title='Track_momentum_matching_tof_and_track',
        xlabel='p [GeV]',
        ylabel='Entries',
        outputname=f'{self.name}/track_momentum',
        rootfile=self.rootfile
    )

    myfunc.make_histogram_root(
        btof_and_track_matched['track_pt'],
        100,
        hist_range=[0, 5],
        title='Track_transverse_momentum_matching_tof_and_track',
        xlabel='pt [GeV]',
        ylabel='Entries',
        outputname=f'{self.name}/track_pt',
        rootfile=self.rootfile
    )

    myfunc.make_histogram_root(
        btof_and_track_matched['track_pathlength'],
        100,
        hist_range=[0, 5000],
        title='Track_pathlength_matching_tof_and_track',
        xlabel='pathlength [mm]',
        ylabel='Entries',
        outputname=f'{self.name}/track_pathlength',
        rootfile=self.rootfile
    )

    myfunc.make_histogram_root(
        btof_and_track_matched['tof_time'],
        100,
        hist_range=[0, 20],
        title='TOF_time_matching_tof_and_track',
        xlabel='time [ns]',
        ylabel='Entries',
        outputname=f'{self.name}/tof_time',
        rootfile=self.rootfile
    )

    myfunc.make_histogram_root(
        ak.flatten(delta_angles_all),
        100,
        hist_range=[0, 0.5],
        title='Delta_angle_all_matching_tof_and_track',
        xlabel='delta_angle [rad]',
        ylabel='Entries',
        outputname=f'{self.name}/delta_angle_all',
        rootfile=self.rootfile
    )

    myfunc.make_histogram_root(
        min_delta_angles_events,
        100,
        hist_range=[0, 0.5],
        title='Min_delta_angle_events_matching_tof_and_track',
        xlabel='delta_angle [rad]',
        ylabel='Entries',
        outputname=f'{self.name}/min_delta_angle_events',
        rootfile=self.rootfile
    )