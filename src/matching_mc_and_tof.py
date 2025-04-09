import numpy as np
import pandas as pd
import awkward as ak

from matching_mc_and_tof_plotter import MatchingMCAndTOFPlotter


class MatchingMCAndTOF:
    def __init__(self, mc, tof, rootfile, name, dis_file, branch):
        self.mc = mc
        self.tof = tof
        self.rootfile = rootfile
        self.name = name
        self.dis_file = dis_file
        self.branch = branch
        self.tof_pid_performance_plotter = MatchingMCAndTOFPlotter(rootfile, name)

    def matching_mc_and_tof(
      self,
      mc_pdg: ak.Array,
      mc_vertex_x: ak.Array,
      mc_vertex_y: ak.Array,
      mc_vertex_z: ak.Array,
      mc_momentum_x: ak.Array,
      mc_momentum_y: ak.Array,
      mc_momentum_z: ak.Array,
      mc_charge: ak.Array,
      mc_generator_status: ak.Array,
      btof_time: ak.Array,
      btof_pos_x: ak.Array,
      btof_pos_y: ak.Array,
      btof_pos_z: ak.Array,
      btof_pos_phi: ak.Array,
      btof_pos_theta: ak.Array,
      btof_pos_r: ak.Array,
      etof_time: ak.Array,
      etof_pos_x: ak.Array,
      etof_pos_y: ak.Array,
      etof_pos_z: ak.Array,
      etof_pos_phi: ak.Array,
      etof_pos_theta: ak.Array,
      etof_pos_r: ak.Array,
      SELECTED_EVENTS: int,
      verbose: bool = False,
      plot_verbose: bool = False
    ):
        """
        Matches Monte Carlo (MC) and Time-of-Flight (TOF) hit information.

        Args:
            mc_pdg (ak.Array): PDG code for MC particles.
            mc_vertex_x (ak.Array): X-coordinate of MC particle vertex.
            mc_vertex_y (ak.Array): Y-coordinate of MC particle vertex.
            mc_vertex_z (ak.Array): Z-coordinate of MC particle vertex.
            mc_momentum_x (ak.Array): X-component of MC particle momentum.
            mc_momentum_y (ak.Array): Y-component of MC particle momentum.
            mc_momentum_z (ak.Array): Z-component of MC particle momentum.
            mc_charge (ak.Array): Charge of MC particle.
            mc_generator_status (ak.Array): Generator status of MC particle.
            btof_time (ak.Array): Time-of-Flight (TOF) hit time.
            btof_pos_x (ak.Array): X-coordinate of TOF hit position.
            btof_pos_y (ak.Array): Y-coordinate of TOF hit position.
            btof_pos_z (ak.Array): Z-coordinate of TOF hit position.
            btof_pos_phi (ak.Array): Phi angle of TOF hit position.
            btof_pos_theta (ak.Array): Theta angle of TOF hit position.
            btof_pos_r (ak.Array): Radial distance of TOF hit position.

        Returns:
            Tuple[dict, dict]: Dictionaries containing MC and TOF hit information.
        """

        btof_hit_info = {
            'event': [],
            'mc_index': [],
            'mc_pdg': [],
            'mc_generator_status': [],
            'mc_charge': [],
            'mc_vertex_x': [],
            'mc_vertex_y': [],
            'mc_vertex_z': [],
            'mc_momentum_x': [],
            'mc_momentum_y': [],
            'mc_momentum_z': [],
            'mc_momentum': [],
            'mc_momentum_phi': [],
            'mc_momentum_theta': [],
            'tof_time': [],
            'tof_pos_x': [],
            'tof_pos_y': [],
            'tof_pos_z': [],
            'tof_pos_phi': [],
            'tof_pos_theta': [],
            'tof_pos_r': [],
        }

        etof_hit_info = {
            'event': [],
            'mc_index': [],
            'mc_pdg': [],
            'mc_generator_status': [],
            'mc_charge': [],
            'mc_vertex_x': [],
            'mc_vertex_y': [],
            'mc_vertex_z': [],
            'mc_momentum_x': [],
            'mc_momentum_y': [],
            'mc_momentum_z': [],
            'mc_momentum': [],
            'mc_momentum_phi': [],
            'mc_momentum_theta': [],
            'tof_time': [],
            'tof_pos_x': [],
            'tof_pos_y': [],
            'tof_pos_z': [],
            'tof_pos_phi': [],
            'tof_pos_theta': [],
            'tof_pos_r': [],
        }

        print("Matching MC and TOF hit information ...")
        btof_mc_index = self.dis_file[self.branch['btof_raw_hit_mc_associaction_branch'][0]].array(library='ak')
        etof_mc_index = self.dis_file[self.branch['etof_raw_hit_mc_associaction_branch'][0]].array(library='ak')

        selected_events_btof = min(SELECTED_EVENTS, len(btof_mc_index))
        selected_events_etof = min(SELECTED_EVENTS, len(etof_mc_index))

        for i in range(selected_events_btof):
            for j in range(len(btof_mc_index[i])):
                btof_hit_info['event'].append(i)
                btof_hit_info['mc_index'].append(btof_mc_index[i][j])
                btof_hit_info['mc_pdg'].append(mc_pdg[i][btof_mc_index[i][j]])
                btof_hit_info['mc_generator_status'].append(mc_generator_status[i][btof_mc_index[i][j]])
                btof_hit_info['mc_charge'].append(mc_charge[i][btof_mc_index[i][j]])
                btof_hit_info['mc_vertex_x'].append(mc_vertex_x[i][btof_mc_index[i][j]])
                btof_hit_info['mc_vertex_y'].append(mc_vertex_y[i][btof_mc_index[i][j]])
                btof_hit_info['mc_vertex_z'].append(mc_vertex_z[i][btof_mc_index[i][j]])
                btof_hit_info['mc_momentum_x'].append(mc_momentum_x[i][btof_mc_index[i][j]])
                btof_hit_info['mc_momentum_y'].append(mc_momentum_y[i][btof_mc_index[i][j]])
                btof_hit_info['mc_momentum_z'].append(mc_momentum_z[i][btof_mc_index[i][j]])
                btof_hit_info['mc_momentum'].append(np.sqrt(mc_momentum_x[i][btof_mc_index[i][j]]**2 + mc_momentum_y[i][btof_mc_index[i][j]]**2 + mc_momentum_z[i][btof_mc_index[i][j]]**2))
                btof_hit_info['mc_momentum_phi'].append(np.arctan2(mc_momentum_y[i][btof_mc_index[i][j]], mc_momentum_x[i][btof_mc_index[i][j]]))
                btof_hit_info['mc_momentum_theta'].append(np.arccos(mc_momentum_z[i][btof_mc_index[i][j]]/np.sqrt(mc_momentum_x[i][btof_mc_index[i][j]]**2 + mc_momentum_y[i][btof_mc_index[i][j]]**2 + mc_momentum_z[i][btof_mc_index[i][j]]**2)))
                btof_hit_info['tof_time'].append(btof_time[i][j])
                btof_hit_info['tof_pos_x'].append(btof_pos_x[i][j])
                btof_hit_info['tof_pos_y'].append(btof_pos_y[i][j])
                btof_hit_info['tof_pos_z'].append(btof_pos_z[i][j])
                btof_hit_info['tof_pos_phi'].append(btof_pos_phi[i][j])
                btof_hit_info['tof_pos_theta'].append(btof_pos_theta[i][j])
                btof_hit_info['tof_pos_r'].append(btof_pos_r[i][j])

                #event progress
                if i % 1000 == 0:
                    print(f"btof Processing event {i} ...")

        for i in range(selected_events_etof):
            for j in range(len(etof_mc_index[i])):
                etof_hit_info['event'].append(i)
                etof_hit_info['mc_index'].append(etof_mc_index[i][j])
                etof_hit_info['mc_pdg'].append(mc_pdg[i][etof_mc_index[i][j]])
                etof_hit_info['mc_generator_status'].append(mc_generator_status[i][etof_mc_index[i][j]])
                etof_hit_info['mc_charge'].append(mc_charge[i][etof_mc_index[i][j]])
                etof_hit_info['mc_vertex_x'].append(mc_vertex_x[i][etof_mc_index[i][j]])
                etof_hit_info['mc_vertex_y'].append(mc_vertex_y[i][etof_mc_index[i][j]])
                etof_hit_info['mc_vertex_z'].append(mc_vertex_z[i][etof_mc_index[i][j]])
                etof_hit_info['mc_momentum_x'].append(mc_momentum_x[i][etof_mc_index[i][j]])
                etof_hit_info['mc_momentum_y'].append(mc_momentum_y[i][etof_mc_index[i][j]])
                etof_hit_info['mc_momentum_z'].append(mc_momentum_z[i][etof_mc_index[i][j]])
                etof_hit_info['mc_momentum'].append(np.sqrt(mc_momentum_x[i][etof_mc_index[i][j]]**2 + mc_momentum_y[i][etof_mc_index[i][j]]**2 + mc_momentum_z[i][etof_mc_index[i][j]]**2))
                etof_hit_info['mc_momentum_phi'].append(np.arctan2(mc_momentum_y[i][etof_mc_index[i][j]], mc_momentum_x[i][etof_mc_index[i][j]]))
                etof_hit_info['mc_momentum_theta'].append(np.arccos(mc_momentum_z[i][etof_mc_index[i][j]]/np.sqrt(mc_momentum_x[i][etof_mc_index[i][j]]**2 + mc_momentum_y[i][etof_mc_index[i][j]]**2 + mc_momentum_z[i][etof_mc_index[i][j]]**2)))
                etof_hit_info['tof_time'].append(etof_time[i][j])
                etof_hit_info['tof_pos_x'].append(etof_pos_x[i][j])
                etof_hit_info['tof_pos_y'].append(etof_pos_y[i][j])
                etof_hit_info['tof_pos_z'].append(etof_pos_z[i][j])
                etof_hit_info['tof_pos_phi'].append(etof_pos_phi[i][j])
                etof_hit_info['tof_pos_theta'].append(etof_pos_theta[i][j])
                etof_hit_info['tof_pos_r'].append(etof_pos_r[i][j])

                #event progress
                if i % 1000 == 0:
                    print(f"etof Processing event {i} ...")

        btof_hit_info_df = pd.DataFrame(btof_hit_info)
        btof_hit_info_df.to_csv(f'./out/{self.name}/btof_hit_info.csv')

        etof_hit_info_df = pd.DataFrame(etof_hit_info)
        etof_hit_info_df.to_csv(f'./out/{self.name}/etof_hit_info.csv')

        if plot_verbose:
            self.tof_pid_performance_plotter.plot_matching_mc_and_tof(btof_hit_info_df, area='btof')
            self.tof_pid_performance_plotter.plot_matching_mc_and_tof(etof_hit_info_df, area='etof')

        print(" completed matching MC and TOF hit information ...")
        return btof_hit_info_df, etof_hit_info_df

    def filtered_stable_particle_hit_and_generated_point(self, btof_hit_info, etof_hit_info,plot_verbose: bool = False):
        """
        Filters stable particle hits.

        Args:
            btof_hit_and_mc_info (dict): Dictionary containing MC and TOF hit information.

        Returns:
            dict: Dictionary containing filtered MC and TOF hit information.
        """

        print("Filtering stable particle hits ...")
        stable_btof_hit_info = {
            'event': [],
            'mc_index': [],
            'mc_pdg': [],
            'mc_generator_status': [],
            'mc_charge': [],
            'mc_vertex_x': [],
            'mc_vertex_y': [],
            'mc_vertex_z': [],
            'mc_momentum_x': [],
            'mc_momentum_y': [],
            'mc_momentum_z': [],
            'mc_momentum': [],
            'mc_momentum_phi': [],
            'mc_momentum_theta': [],
            'time': [],
            'position_x': [],
            'position_y': [],
            'position_z': [],
        }

        stable_etof_hit_info = {
            'event': [],
            'mc_index': [],
            'mc_pdg': [],
            'mc_generator_status': [],
            'mc_charge': [],
            'mc_vertex_x': [],
            'mc_vertex_y': [],
            'mc_vertex_z': [],
            'mc_momentum_x': [],
            'mc_momentum_y': [],
            'mc_momentum_z': [],
            'mc_momentum': [],
            'mc_momentum_phi': [],
            'mc_momentum_theta': [],
            'time': [],
            'position_x': [],
            'position_y': [],
            'position_z': [],
        }

        for i in range(len(btof_hit_info)):
            if btof_hit_info['mc_generator_status'][i] == 1 and btof_hit_info['mc_charge'][i] != 0 and btof_hit_info['mc_vertex_z'][i] > -5 and btof_hit_info['mc_vertex_z'][i] < 5:
            # if btof_hit_info['mc_generator_status'][i] == 1 and btof_hit_info['mc_charge'][i] != 0:
                stable_btof_hit_info['event'].append(btof_hit_info['event'][i])
                stable_btof_hit_info['mc_index'].append(btof_hit_info['mc_index'][i])
                stable_btof_hit_info['mc_pdg'].append(btof_hit_info['mc_pdg'][i])
                stable_btof_hit_info['mc_generator_status'].append(btof_hit_info['mc_generator_status'][i])
                stable_btof_hit_info['mc_charge'].append(btof_hit_info['mc_charge'][i])
                stable_btof_hit_info['mc_vertex_x'].append(btof_hit_info['mc_vertex_x'][i])
                stable_btof_hit_info['mc_vertex_y'].append(btof_hit_info['mc_vertex_y'][i])
                stable_btof_hit_info['mc_vertex_z'].append(btof_hit_info['mc_vertex_z'][i])
                stable_btof_hit_info['mc_momentum_x'].append(btof_hit_info['mc_momentum_x'][i])
                stable_btof_hit_info['mc_momentum_y'].append(btof_hit_info['mc_momentum_y'][i])
                stable_btof_hit_info['mc_momentum_z'].append(btof_hit_info['mc_momentum_z'][i])
                stable_btof_hit_info['mc_momentum'].append(btof_hit_info['mc_momentum'][i])
                stable_btof_hit_info['mc_momentum_phi'].append(btof_hit_info['mc_momentum_phi'][i])
                stable_btof_hit_info['mc_momentum_theta'].append(btof_hit_info['mc_momentum_theta'][i])
                stable_btof_hit_info['time'].append(btof_hit_info['tof_time'][i])
                stable_btof_hit_info['position_x'].append(btof_hit_info['tof_pos_x'][i])
                stable_btof_hit_info['position_y'].append(btof_hit_info['tof_pos_y'][i])
                stable_btof_hit_info['position_z'].append(btof_hit_info['tof_pos_z'][i])

        for i in range(len(etof_hit_info)):
            if etof_hit_info['mc_generator_status'][i] == 1 and etof_hit_info['mc_charge'][i] != 0 and etof_hit_info['mc_vertex_z'][i] > -5 and etof_hit_info['mc_vertex_z'][i] < 5:
            # if etof_hit_info['mc_generator_status'][i] == 1 and etof_hit_info['mc_charge'][i] != 0:
                stable_etof_hit_info['event'].append(etof_hit_info['event'][i])
                stable_etof_hit_info['mc_index'].append(etof_hit_info['mc_index'][i])
                stable_etof_hit_info['mc_pdg'].append(etof_hit_info['mc_pdg'][i])
                stable_etof_hit_info['mc_generator_status'].append(etof_hit_info['mc_generator_status'][i])
                stable_etof_hit_info['mc_charge'].append(etof_hit_info['mc_charge'][i])
                stable_etof_hit_info['mc_vertex_x'].append(etof_hit_info['mc_vertex_x'][i])
                stable_etof_hit_info['mc_vertex_y'].append(etof_hit_info['mc_vertex_y'][i])
                stable_etof_hit_info['mc_vertex_z'].append(etof_hit_info['mc_vertex_z'][i])
                stable_etof_hit_info['mc_momentum_x'].append(etof_hit_info['mc_momentum_x'][i])
                stable_etof_hit_info['mc_momentum_y'].append(etof_hit_info['mc_momentum_y'][i])
                stable_etof_hit_info['mc_momentum_z'].append(etof_hit_info['mc_momentum_z'][i])
                stable_etof_hit_info['mc_momentum'].append(etof_hit_info['mc_momentum'][i])
                stable_etof_hit_info['mc_momentum_phi'].append(etof_hit_info['mc_momentum_phi'][i])
                stable_etof_hit_info['mc_momentum_theta'].append(etof_hit_info['mc_momentum_theta'][i])
                stable_etof_hit_info['time'].append(etof_hit_info['tof_time'][i])
                stable_etof_hit_info['position_x'].append(etof_hit_info['tof_pos_x'][i])
                stable_etof_hit_info['position_y'].append(etof_hit_info['tof_pos_y'][i])
                stable_etof_hit_info['position_z'].append(etof_hit_info['tof_pos_z'][i])

        stable_btof_hit_info_df = pd.DataFrame(stable_btof_hit_info)
        stable_btof_hit_info_df.to_csv(f'./out/{self.name}/stable_particle_btof_hit.csv')

        stable_etof_hit_info_df = pd.DataFrame(stable_etof_hit_info)
        stable_etof_hit_info_df.to_csv(f'./out/{self.name}/stable_particle_etof_hit.csv')

        print(" completed filtering stable particle hits ...")

        if plot_verbose:
            self.tof_pid_performance_plotter.plot_filtered_stable_particle_hit_and_generated_point(stable_btof_hit_info_df, area='btof')
            self.tof_pid_performance_plotter.plot_filtered_stable_particle_hit_and_generated_point(stable_etof_hit_info_df, area='etof')

        return stable_btof_hit_info_df, stable_etof_hit_info_df

    def isReconstructedHit(self, stable_btof_hit_info_df, stable_etof_hit_info_df):
        """
        Checks if a hit is reconstructed.

        Args:
            btof_hit_and_mc_info (dict): Dictionary containing MC and TOF hit information.
            stable_particle_hit (dict): Dictionary containing filtered MC and TOF hit information.

        Returns:
            dict: Dictionary containing reconstructed MC and TOF hit information.
        """
        
        # stable_btof_hit_info_df = pd.read_csv(f'./out/{self.name}/stable_particle_btof_hit.csv')
        # stable_etof_hit_info_df = pd.read_csv(f'./out/{self.name}/stable_particle_etof_hit.csv')
        print("Filtering stable particle hits ...")

        filtered_rows_btof = []
        filtered_rows_etof = []

        unique_events_btof = stable_btof_hit_info_df['event'].unique()
        unique_events_etof = stable_etof_hit_info_df['event'].unique()


        btof_rec_position_x = self.dis_file[self.branch['new_btof_rec_hit_branch'][1]].array(library='ak')
        # btof_rec_position_x = self.dis_file[self.branch['old_btof_rec_hit_branch'][1]].array(library='ak')

        etof_rec_position_x = self.dis_file[self.branch['new_etof_rec_hit_branch'][1]].array(library='ak')

        for event in unique_events_btof:
            event_df = stable_btof_hit_info_df[stable_btof_hit_info_df['event'] == event]
            
            new_array = event_df['position_x'].values.astype(float)
            
            rec_array = np.array(btof_rec_position_x[event], dtype=float)

            print(f"Processing event {event} ...")
            
            matching_new_indices = []
            for x in rec_array:
                idx = np.where(np.isclose(new_array, x, atol=1e-1))[0]
                if idx.size > 0:
                    closest_idx = idx[np.argmin(np.abs(new_array[idx] - x))]
                    matching_new_indices.append(closest_idx)
            
            filtered_event_df = event_df.iloc[matching_new_indices]
            
            filtered_rows_btof.append(filtered_event_df)

        for event in unique_events_etof:
            event_df = stable_etof_hit_info_df[stable_etof_hit_info_df['event'] == event]
            
            new_array = event_df['position_x'].values.astype(float)
            
            rec_array = np.array(etof_rec_position_x[event], dtype=float)

            print(f"Processing event {event} ...")
            
            matching_new_indices = []
            for x in rec_array:
                idx = np.where(np.isclose(new_array, x, atol=1e-1))[0]
                if idx.size > 0:
                    closest_idx = idx[np.argmin(np.abs(new_array[idx] - x))]
                    matching_new_indices.append(closest_idx)
            
            filtered_event_df = event_df.iloc[matching_new_indices]
            
            filtered_rows_etof.append(filtered_event_df)

        filtered_stable_btof_hit_info = pd.concat(filtered_rows_btof, ignore_index=True)
        filtered_stable_btof_hit_info.to_csv(f'./out/{self.name}/filtered_stable_btof_hit_info.csv', index=False)

        filtered_stable_etof_hit_info = pd.concat(filtered_rows_etof, ignore_index=True)
        filtered_stable_etof_hit_info.to_csv(f'./out/{self.name}/filtered_stable_etof_hit_info.csv', index=False)


        print(" completed filtering stable particle hits ...")

        return filtered_stable_btof_hit_info, filtered_stable_etof_hit_info