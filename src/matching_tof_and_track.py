import pandas as pd
import numpy as np

from utility_function import angular_distance
from matching_tof_and_track_plotter import MatchingTOFAndTrackPlotter

class MatchingTOFAndTrack:
    def __init__(self, btof, track, rootfile, name: str):
        """
        Constructor for the MatchingTOFAndTrack class."
        """
        self.name = name
        self.rootfile = rootfile
        self.btof = btof
        self.track = track
        self.matching_tof_and_track_plotter = MatchingTOFAndTrackPlotter(rootfile, name)

    def matching_tof_and_track(
      self,
      track_segments_on_btof_df: pd.DataFrame,
      filtered_stable_btof_hit_info: pd.DataFrame,
      verbose: bool = False,
      plot_verbose: bool = False,
    ):

        track_segments_on_btof_pd = pd.read_csv(f'./out/{self.name}/track_segments_on_btof.csv')
        filtered_stable_btof_hit_info = pd.read_csv(f'./out/{self.name}/filtered_stable_btof_hit_info.csv')

        print("Matching BTOF and track segments ...")

        btof_and_track_matched = {
        'event_idx': [],
        'track_idx': [],
        'track_p': [],
        'track_pt': [],
        'track_pos_phi': [],
        'track_pos_theta': [],
        'track_pos_x': [],
        'track_pos_y': [],
        'track_pos_z': [],
        'tof_pos_phi': [],
        'tof_pos_theta': [],
        'tof_time': [],
        'mc_pdg': [],
        'mc_momentum': [],
        'mc_vertex_x': [],
        'mc_vertex_y': [],
        'mc_vertex_z': [],
        'track_pathlength': [],           
        'delta_angle': [],
        }

        min_delta_angles_events = []
        delta_angles_all = []

        angle_threshold = 0.2

        for i in range(len(track_segments_on_btof_pd)):

            print(f'Processing track: {i} / {len(track_segments_on_btof_pd)}')

            event_idx = track_segments_on_btof_pd['event'][i]
            
            track_pos_x       = float(track_segments_on_btof_pd['track_pos_x'].iloc[i])
            track_pos_y       = float(track_segments_on_btof_pd['track_pos_y'].iloc[i])
            track_pos_z       = float(track_segments_on_btof_pd['track_pos_z'].iloc[i])
            track_pos_phi     = float(np.arctan2(track_segments_on_btof_pd['track_pos_y'].iloc[i],
                                            track_segments_on_btof_pd['track_pos_x'].iloc[i]))
            track_pos_theta   = float(np.arccos(track_segments_on_btof_pd['track_pos_z'].iloc[i] /
                                np.sqrt(track_segments_on_btof_pd['track_pos_x'].iloc[i]**2 +
                                        track_segments_on_btof_pd['track_pos_y'].iloc[i]**2 +
                                        track_segments_on_btof_pd['track_pos_z'].iloc[i]**2)))
            track_phi         = float(track_segments_on_btof_pd['track_momentum_phi'].iloc[i])
            track_theta       = float(track_segments_on_btof_pd['track_momentum_theta'].iloc[i])
            track_pathlength  = float(track_segments_on_btof_pd['track_pathlength'].iloc[i])
            track_momentum    = float(track_segments_on_btof_pd['track_momentum'].iloc[i])
            track_momentum_pt = float(track_segments_on_btof_pd['track_momentum_pt'].iloc[i])
            
            tof_subset        = filtered_stable_btof_hit_info[filtered_stable_btof_hit_info['event'] == event_idx]
            tof_times         = tof_subset['time']
            tof_pos_phi       = np.arctan2(tof_subset['position_y'], tof_subset['position_x'])
            tof_pos_theta     = np.arccos(tof_subset['position_z'] /
                                    np.sqrt(tof_subset['position_x']**2 +
                                            tof_subset['position_y']**2 +
                                            tof_subset['position_z']**2))
            
            tof_pos_phis_array = np.array(tof_pos_phi)
            tof_pos_thetas_array = np.array(tof_pos_theta)

            mc_subset         = tof_subset  
            mc_pdg            = mc_subset['mc_pdg']
            mc_momentum       = mc_subset['mc_momentum']
            mc_vertex_x       = mc_subset['mc_vertex_x']
            mc_vertex_y       = mc_subset['mc_vertex_y']
            mc_vertex_z       = mc_subset['mc_vertex_z']

            if tof_pos_phis_array.size == 0:
                continue

            # if track_momentum < 0.1:
            #     continue

            delta_angles = angular_distance(
                track_pos_phi,    
                track_pos_theta,  
                tof_pos_phis_array, 
                tof_pos_thetas_array  
            )
            delta_angles_all.append(delta_angles)
            min_idx = np.argmin(delta_angles)
            min_delta_angle = delta_angles[min_idx]

            if min_delta_angle < angle_threshold:
                    btof_and_track_matched['event_idx'].append(event_idx)
                    track_idx = track_segments_on_btof_pd['track_segments_id'].iloc[i]
                    btof_and_track_matched['track_idx'].append(track_idx)
                    btof_and_track_matched['track_p'].append(track_momentum)
                    btof_and_track_matched['track_pt'].append(track_momentum_pt)
                    btof_and_track_matched['track_pos_phi'].append(track_phi)
                    btof_and_track_matched['track_pos_theta'].append(track_theta)
                    btof_and_track_matched['track_pos_x'].append(track_pos_x)
                    btof_and_track_matched['track_pos_y'].append(track_pos_y)
                    btof_and_track_matched['track_pos_z'].append(track_pos_z)
                    btof_and_track_matched['tof_pos_phi'].append(tof_pos_phis_array[min_idx])
                    btof_and_track_matched['tof_pos_theta'].append(tof_pos_thetas_array[min_idx])
                    btof_and_track_matched['tof_time'].append(tof_times.iloc[min_idx])
                    btof_and_track_matched['mc_pdg'].append(mc_pdg.iloc[min_idx])
                    btof_and_track_matched['mc_momentum'].append(mc_momentum.iloc[min_idx])
                    btof_and_track_matched['mc_vertex_x'].append(mc_vertex_x.iloc[min_idx])
                    btof_and_track_matched['mc_vertex_y'].append(mc_vertex_y.iloc[min_idx])
                    btof_and_track_matched['mc_vertex_z'].append(mc_vertex_z.iloc[min_idx])
                    btof_and_track_matched['track_pathlength'].append(track_pathlength)
                    btof_and_track_matched['delta_angle'].append(min_delta_angle)
                    min_delta_angles_events.append(min_delta_angle)

        btof_and_track_matched_df = pd.DataFrame(btof_and_track_matched)
        btof_and_track_matched_df.to_csv(f'./out/{self.name}/btof_and_track_matched.csv', index=False)

        if plot_verbose:
            self.matching_tof_and_track_plotter.plot_matched_tof_and_track(
                btof_and_track_matched_df,
                min_delta_angles_events,
                delta_angles_all
            )

        print(f"Number of matched BTOF and track segments: {len(btof_and_track_matched_df)}")

        return btof_and_track_matched_df
