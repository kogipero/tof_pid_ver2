import uproot
import numpy as np
import ROOT as r
import argparse
import sys
import os

from typing import List, Tuple

current_dir = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from utility_function import load_yaml_config, load_tree_file, make_directory
from track_reader import TrackReader
from mc_reader import MCReader
from tof_reader import TOFReader
from matching_mc_and_tof import MatchingMCAndTOF
from matching_tof_and_track import MatchingTOFAndTrack
from tof_pid_performance_manager import ToFPIDPerformanceManager

def analyzer(
    name: str,
    rootfile: uproot.TTree,
    output_txt_name: str = 'pid_result.txt',
    output_efficiency_result_name: str = 'matching_result.txt'
):
    """
    Executes the full track and MC matching analysis with PID performance evaluation.
    """
    config = load_yaml_config('./config/execute_config.yaml')
    branch = load_yaml_config('./config/branch_name.yaml')
    file_path = load_yaml_config('./config/file_path.yaml')

    name = config['directory_name']
    VERBOSE = config['VERBOSE']
    PLOT_VERBOSE = config['PLOT_VERBOSE']
    SELECTED_EVENTS = config['SELECTED_EVENTS']        
    analysis_event_type = config['analysis_event_type']  

    filename = file_path[analysis_event_type]['path']
    tree = load_tree_file(filename)

    tof = TOFReader(dis_file=tree, branch=branch, name=name, rootfile=rootfile)

    btof_pos_x, btof_pos_y, btof_pos_z, btof_time, btof_phi, btof_theta, btof_r, ectof_pos_x, ectof_pos_y, ectof_pos_z, ectof_time, ectof_phi, ectof_theta, ectof_r = tof.get_tof_info(
        name, SELECTED_EVENTS, rootfile=rootfile, verbose=VERBOSE, plot_verbose=PLOT_VERBOSE
    )

    mc = MCReader(dis_file=tree, branch=branch, name=name, rootfile=rootfile)
    mc_px, mc_py, mc_pz, mc_p, mc_p_theta, mc_p_phi, mc_pdg, mc_charge, mc_generatorStatus, mc_vertex_x, mc_vertex_y, mc_vertex_z = mc.get_mc_info(
        verbose=VERBOSE, plot_verbose=PLOT_VERBOSE
    )

    match_mc_and_tof = MatchingMCAndTOF(mc=mc, tof=tof, rootfile=rootfile, name=name, dis_file=tree, branch=branch)
    btof_hit_and_mc_info = match_mc_and_tof.matching_mc_and_tof(
        mc_pdg, mc_vertex_x, mc_vertex_y, mc_vertex_z, mc_px, mc_py, mc_pz, mc_charge, mc_generatorStatus,
        btof_time, btof_pos_x, btof_pos_y, btof_pos_z, btof_phi, btof_theta, btof_r, SELECTED_EVENTS, VERBOSE, PLOT_VERBOSE
    )

    stable_particle_hit = match_mc_and_tof.filtered_stable_particle_hit_and_generated_point(btof_hit_and_mc_info, PLOT_VERBOSE)
    filtered_stable_btof_hit_info = match_mc_and_tof.isReconstructedHit(stable_particle_hit)

    track = TrackReader(dis_file=tree, branch=branch, name=name, rootfile=rootfile)

    # Retrieve track segments positions and momenta
    track_segments_x, track_segments_y, track_segments_z, _, _ = track.get_track_segments_pos(
        name=name, rootfile=rootfile, verbose=VERBOSE, plot_verbose=PLOT_VERBOSE
    )
    track_segments_px, track_segments_py, track_segments_pz, track_segments_p, tracksegments_pt, track_segments_p_theta, track_segments_p_phi, track_segment_pathlength = track.get_track_segments_momentum(
        name=name, rootfile=rootfile, verbose=VERBOSE, plot_verbose=PLOT_VERBOSE
    )

    # Split track segments into individual tracks
    all_tracks = track.split_track_segments(
        x_positions=track_segments_x,
        y_positions=track_segments_y,
        z_positions=track_segments_z,
        px_momenta=track_segments_px,
        py_momenta=track_segments_py,
        pz_momenta=track_segments_pz,
        track_segment_pathlength=track_segment_pathlength,
        margin_theta=0.6,
        margin_phi=0.6,
        rootfile=rootfile,
        verbose=VERBOSE,
        plot_verbose=PLOT_VERBOSE,
        SELECTED_EVENTS=SELECTED_EVENTS
    )

    track_segments_on_btof_df = track.get_track_segments_on_tof_info(all_tracks, name, rootfile, verbose=VERBOSE, plot_verbose=PLOT_VERBOSE)

    match_tof_and_track = MatchingTOFAndTrack(btof=tof, track=track, rootfile=rootfile, name=name)
    btof_and_track_matched_df = match_tof_and_track.matching_tof_and_track(
        track_segments_on_btof_df, filtered_stable_btof_hit_info, verbose=VERBOSE, plot_verbose=PLOT_VERBOSE
    )

    pid = ToFPIDPerformanceManager(dis_file=tree, branch=branch, name=name, rootfile=rootfile)
    btof_calc_mass, btof_pdg, track_momentums_on_btof, track_momentums_transverse_on_btof = pid.process_pid_performance_plot(
        name, btof_and_track_matched_df, rootfile=rootfile,
        output_txt_name=output_txt_name, plot_verbose=PLOT_VERBOSE
    )

    bin_center, separation_power = pid.process_separation_power_vs_momentum(
        btof_calc_mass=btof_calc_mass,
        btof_pdg=btof_pdg,
        track_momentums_on_btof=track_momentums_on_btof,
        track_momentums_transverse_on_btof=track_momentums_transverse_on_btof,
        plot_verbose=PLOT_VERBOSE
    )

    pid.process_purity_vs_momentum(
        btof_calc_mass=btof_calc_mass,
        btof_pdg=btof_pdg,
        track_momentums_on_btof=track_momentums_on_btof,
        track_momentums_transverse_on_btof=track_momentums_transverse_on_btof,
        plot_verbose=PLOT_VERBOSE
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run track-MC matching and ToF PID analysis.")
    parser.add_argument("--rootfile", type=str, required=True, help="Output ROOT file name")
    args = parser.parse_args()

    config = load_yaml_config('./config/execute_config.yaml')
    name = config['directory_name']  
    directory_name = f'./out/{name}'
    make_directory(directory_name)

    rootfile_path = os.path.join(directory_name, args.rootfile)

    rootfile = r.TFile(rootfile_path, "RECREATE")
    analyzer(name, rootfile)

    rootfile.Close()
    print("Analysis completed.")
