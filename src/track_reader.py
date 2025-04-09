import uproot
import numpy as np
import awkward as ak
import ROOT as r
import pandas as pd

from typing import Tuple
from typing import List
from track_plotter import TrackPlotter

class TrackReader:
  def __init__(self, dis_file: uproot.TTree, branch: dict, name: str, rootfile: uproot.TTree):
      """
      Constructor for the TrackReader class.

      Args:
          dis_file (uproot.TTree): The ROOT file containing the track information.
          branch (dict): Dictionary containing the branch names.
          name (str): Name for plotting output files.
      """
      self.dis_file = dis_file
      self.branch = branch
      self.name = name
      self.rootfile = rootfile
      self.track_plotter = TrackPlotter(rootfile, name)

  def get_track_segments_pos(
          self,
          name: str,
          rootfile: uproot.TTree = None, 
          verbose: bool = False, 
          plot_verbose: bool = False
      ) -> Tuple[ak.Array, ak.Array, ak.Array, np.ndarray, np.ndarray]:
      """
      Retrieves track segment positions.

      Args:
          name (str): Name for plotting output files.
          verbose (bool): Flag for printing debug information.
          plot_verbose (bool): Flag for generating plots.

      Returns:
          ak.Array, ak.Array, ak.Array, np.ndarray, np.ndarray: Track segment positions and derived quantities.
      """
      print('Start getting track segments')
      track_segments_pos_x = self.dis_file[self.branch['track_branch'][2]].array(library = 'ak') # 2 is the track segment position x branch
      track_segments_pos_y = self.dis_file[self.branch['track_branch'][3]].array(library = 'ak') # 3 is the track segment position y branch
      track_segments_pos_z = self.dis_file[self.branch['track_branch'][4]].array(library = 'ak') # 4 is the track segment position z branch
      track_segments_pos_d = np.sqrt(track_segments_pos_x**2 + track_segments_pos_y**2 + track_segments_pos_z**2)
      track_segments_pos_r = np.sqrt(track_segments_pos_x**2 + track_segments_pos_y**2)

      if verbose:
          print(f"Number of track segments: {len(track_segments_pos_x)}")

      if plot_verbose:
          self.track_plotter.plot_track_segments_pos(track_segments_pos_x, track_segments_pos_y, track_segments_pos_z, track_segments_pos_d, track_segments_pos_r)
          
      print('End getting track segments')

      return track_segments_pos_x, track_segments_pos_y, track_segments_pos_z, track_segments_pos_d, track_segments_pos_r

  def get_track_segments_momentum(
              self, 
              name: str, 
              rootfile: uproot.TTree,
              verbose: bool = False, 
              plot_verbose: bool = False
          ) -> Tuple[ak.Array, ak.Array, ak.Array, np.ndarray, np.ndarray, np.ndarray, ak.Array]:
          """
          Retrieves track segment momentum and computes derived quantities(pathlength).

          Args:
              name (str): Name for plotting output files.
              verbose (bool): Flag for printing debug information.
              plot_verbose (bool): Flag for generating plots.

          Returns:
              Tuple[ak.Array, ak.Array, ak.Array, np.ndarray, np.ndarray, np.ndarray, ak.Array]: Track segment momenta and derived quantities.
          """
          print('Start getting track segments momentum')

          track_segments_px = self.dis_file[self.branch['track_branch'][11]].array(library='ak') # 11 is the index of the px branch
          track_segments_py = self.dis_file[self.branch['track_branch'][12]].array(library='ak') # 12 is the index of the py branch
          track_segments_pz = self.dis_file[self.branch['track_branch'][13]].array(library='ak') # 13 is the index of the pz branch
          track_segments_p = np.sqrt(track_segments_px**2 + track_segments_py**2 + track_segments_pz**2)
          track_segments_pt = np.sqrt(track_segments_px**2 + track_segments_py**2)
          track_segments_p_theta = np.where(track_segments_p != 0, np.arccos(track_segments_pz / track_segments_p), 0)
          track_segments_p_phi = np.arctan2(track_segments_py, track_segments_px)
          track_segment_pathlength = self.dis_file[self.branch['track_branch'][27]].array(library='ak') # 27 is the index of the pathlength branch

          if verbose:
              print(f"Number of track segments: {len(track_segments_px)}")

          if plot_verbose:
              self.track_plotter.plot_track_segments_momentum(track_segments_px, track_segments_py, track_segments_pz, track_segments_p, track_segments_pt, track_segments_p_theta, track_segments_p_phi, track_segment_pathlength)
              
          print('End getting track segments momentum')

          return track_segments_px, track_segments_py, track_segments_pz, track_segments_p, track_segments_pt, track_segments_p_theta, track_segments_p_phi, track_segment_pathlength

  def split_track_segments(
          self, 
          x_positions: ak.Array, 
          y_positions: ak.Array, 
          z_positions: ak.Array,
          px_momenta: ak.Array, 
          py_momenta: ak.Array, 
          pz_momenta: ak.Array, 
          track_segment_pathlength: ak.Array, 
          margin_theta: float, 
          margin_phi: float, 
          rootfile: uproot.TTree,
          verbose: bool = False, 
          plot_verbose: bool = False, 
          SELECTED_EVENTS: int = 50000
      ) -> List[List[List[Tuple]]]:
      """
      Splits track segments into individual tracks based on angular separation.

      Args:
          x_positions (ak.Array): X positions of track segments.
          y_positions (ak.Array): Y positions of track segments.
          z_positions (ak.Array): Z positions of track segments.
          px_momentum (ak.Array): Px components of track momentum.
          py_momentum (ak.Array): Py components of track momentum.
          pz_momentum (ak.Array): Pz components of track momentum.
          track_segment_pathlength (ak.Array): Path lengths of track segments.
          margin_theta (float): Angular margin for splitting tracks in theta.
          margin_phi (float): Angular margin for splitting tracks in phi.
          verbose (bool): Flag for printing debug information.
          plot_verbose (bool): Flag for generating plots.
          SELECTED_EVENTS (int): Number of events to process.

      Returns:
          List[List[List[Tuple]]]: Nested list of tracks for each event.
      """
      print('Start splitting track segments')

      all_tracks = []
      selected_events = min(SELECTED_EVENTS, len(x_positions))

      for event in range(selected_events):
          if len(x_positions[event]) == 0:
              print(f"Skipping empty event at index {event}")
              continue

          event_tracks = []
          r = np.sqrt(x_positions[event][0]**2 + y_positions[event][0]**2)
          current_track = [(event, 
                            0, 
                            0, 
                            x_positions[event][0], 
                            y_positions[event][0], 
                            z_positions[event][0],
                            px_momenta[event][0], 
                            py_momenta[event][0], 
                            pz_momenta[event][0], 
                            r, 
                            track_segment_pathlength[event][0])
                            ]

          for i in range(1, len(x_positions[event])):
              px1, py1, pz1 = px_momenta[event][i - 1], py_momenta[event][i - 1], pz_momenta[event][i - 1]
              px2, py2, pz2 = px_momenta[event][i], py_momenta[event][i], pz_momenta[event][i]

              theta1, phi1 = np.arctan2(np.sqrt(px1**2 + py1**2), pz1), np.arctan2(py1, px1)
              theta2, phi2 = np.arctan2(np.sqrt(px2**2 + py2**2), pz2), np.arctan2(py2, px2)

              if abs(theta2 - theta1) < margin_theta and abs(phi2 - phi1) < margin_phi:
                  r = np.sqrt(x_positions[event][i]**2 + y_positions[event][i]**2)
                  z = z_positions[event][i]

                  current_track.append((event,
                                        i, 
                                        len(event_tracks), 
                                        x_positions[event][i], 
                                        y_positions[event][i], 
                                        z_positions[event][i], 
                                        px_momenta[event][i], 
                                        py_momenta[event][i], 
                                        pz_momenta[event][i], 
                                        r, 
                                        track_segment_pathlength[event][i])
                                        )

              else:
                  if current_track:
                      event_tracks.append(current_track)

                  r = np.sqrt(x_positions[event][i]**2 + y_positions[event][i]**2)

                  current_track = [(event,
                                    i, 
                                    len(event_tracks), 
                                    x_positions[event][i], 
                                    y_positions[event][i], 
                                    z_positions[event][i], 
                                    px_momenta[event][i], 
                                    py_momenta[event][i], 
                                    pz_momenta[event][i], 
                                    r, 
                                    track_segment_pathlength[event][i])
                                    ]

          if current_track:
              event_tracks.append(current_track)
          all_tracks.append(event_tracks)

          if event % 1000 == 0:
              print(f'Processed event: {event} / {selected_events}')

      if verbose:
          for event_idx, event_tracks in enumerate(all_tracks[:5]):
              print(f'Event {event_idx+1} has {len(event_tracks)} tracks')
                                  
      if plot_verbose:
          self.track_plotter.plot_split_tracks(all_tracks)
    
      print('End splitting track segments')

      all_tracks_df = pd.DataFrame(all_tracks)
      all_tracks_df.to_csv(f'./out/{self.name}/all_tracks.csv')

      return all_tracks

  def get_track_segments_on_tof_info(
        self,
        all_tracks: List[List[List[Tuple]]],
        name: str, 
        rootfile: uproot.TTree, 
        verbose: bool = False, 
        plot_verbose: bool = False
        ): 
        """
        Retrieves track segment information.

        Args:
            name (str): Name for plotting output files.
            verbose (bool): Flag for printing debug information.
            plot_verbose (bool): Flag for generating plots.

        Returns:
            Tuple[ak.Array, ak.Array, ak.Array, np.ndarray, np.ndarray, np.ndarray, ak.Array]: Track segment information.
        """
        
        track_segments_on_btof = {
        'event': [],
        'track_pos_x': [],
        'track_pos_y': [],
        'track_pos_z': [],
        'track_pos_r': [],
        'track_momentum_x': [],
        'track_momentum_y': [],
        'track_momentum_z': [],
        'track_momentum': [],
        'track_momentum_pt': [],
        'track_momentum_phi': [],
        'track_momentum_theta': [],
        'track_pathlength': [],
        'track_segments_id': [],
        }

        missing_matched = {
        'event': [],
        'track_pos_x': [],
        'track_pos_y': [],
        'track_pos_z': [],
        'track_pos_r': [],
        'track_momentum_x': [],
        'track_momentum_y': [],
        'track_momentum_z': [],
        'track_momentum': [],
        'track_momentum_pt': [],
        'track_momentum_phi': [],
        'track_momentum_theta': [],
        'track_pathlength': [],
        'track_segments_id': [],
        }

        for event in range(len(all_tracks)):
            for track in all_tracks[event]:
                for segment in track:
                    if 625 < segment[9] < 660 and -1150 < segment[5] < 1740:
                        # if segment[10] > 500: 
                        track_segments_on_btof['event'].append(segment[0])
                        track_segments_on_btof['track_pos_x'].append(segment[3])
                        track_segments_on_btof['track_pos_y'].append(segment[4])
                        track_segments_on_btof['track_pos_z'].append(segment[5])
                        track_segments_on_btof['track_pos_r'].append(segment[9])
                        track_segments_on_btof['track_momentum_x'].append(segment[6])
                        track_segments_on_btof['track_momentum_y'].append(segment[7])
                        track_segments_on_btof['track_momentum_z'].append(segment[8])
                        track_segments_on_btof['track_momentum'].append(np.sqrt(segment[6]**2 + segment[7]**2 + segment[8]**2))
                        track_segments_on_btof['track_momentum_pt'].append(np.sqrt(segment[6]**2 + segment[7]**2))
                        track_segments_on_btof['track_momentum_phi'].append(np.arctan2(segment[7], segment[6]))
                        track_segments_on_btof['track_momentum_theta'].append(np.arccos(segment[8]/np.sqrt(segment[6]**2 + segment[7]**2 + segment[8]**2)))
                        track_segments_on_btof['track_pathlength'].append(segment[10])
                        track_segments_on_btof['track_segments_id'].append(segment[1])

                        if 0 < segment[10] < 100:
                            missing_matched['event'].append(segment[0])
                            missing_matched['track_pos_x'].append(segment[3])
                            missing_matched['track_pos_y'].append(segment[4])
                            missing_matched['track_pos_z'].append(segment[5])
                            missing_matched['track_pos_r'].append(segment[9])
                            missing_matched['track_momentum_x'].append(segment[6])
                            missing_matched['track_momentum_y'].append(segment[7])
                            missing_matched['track_momentum_z'].append(segment[8])
                            missing_matched['track_momentum'].append(np.sqrt(segment[6]**2 + segment[7]**2 + segment[8]**2))
                            missing_matched['track_momentum_pt'].append(np.sqrt(segment[6]**2 + segment[7]**2))
                            missing_matched['track_momentum_phi'].append(np.arctan2(segment[7], segment[6]))
                            missing_matched['track_momentum_theta'].append(np.arccos(segment[8]/np.sqrt(segment[6]**2 + segment[7]**2 + segment[8]**2)))
                            missing_matched['track_pathlength'].append(segment[10])
                            missing_matched['track_segments_id'].append(segment[1])

        track_segments_on_btof_df = pd.DataFrame(track_segments_on_btof)
        track_segments_on_btof_df.to_csv(f'./out/{self.name}/track_segments_on_btof.csv')

        if plot_verbose:
            self.track_plotter.plot_track_segments_on_tof_info(track_segments_on_btof_df)
            self.track_plotter.plot_missing_matched_track_segments_on_tof_info(missing_matched)

        return track_segments_on_btof_df