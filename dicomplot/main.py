import sys
import argparse
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import logging

# Import ticker module for setting tick locators
import matplotlib.ticker as ticker


logger = logging.getLogger(__name__)


def get_spot_map(dicom_object, ion_beam_sequence_index=0, ion_control_point_sequence_index=0):
    """Get the spot map from the dicom object.

    """
    logger.info(
        f"Getting spot map from dicom object {ion_beam_sequence_index}, {ion_control_point_sequence_index}")
    ibs = dicom_object.IonBeamSequence[ion_beam_sequence_index]
    cps = ibs.IonControlPointSequence[ion_control_point_sequence_index]

    scan_spot_positions = cps.ScanSpotPositionMap
    scan_spot_meter_set_weights = cps.ScanSpotMetersetWeights
    energy = getattr(cps, 'NominalBeamEnergy', -1)

    return scan_spot_positions, scan_spot_meter_set_weights, energy


def plot_map(field_index, energy_layer_index, maps, ax, cbar, fig, max_weight):
    """Plot the selected map and update the color bar if necessary."""
    ax.clear()
    positions, weights, energy = maps[field_index][energy_layer_index]
    x = np.array(positions[0::2]) / 10  # X coordinates
    y = np.array(positions[1::2]) / 10  # Y coordinates

    # Normalize spot sizes: This might need adjustment depending on the desired appearance
    sizes = np.array(weights) * 100 / max_weight  # Example scaling factor

    scatter = ax.scatter(x, y, c=weights, cmap='nipy_spectral', s=sizes,
                         vmin=0, vmax=max_weight, edgecolors='black', linewidths=0.5, alpha=0.7)

    ax.set_aspect('equal')

    # Update the color bar
    if cbar:
        cbar.update_normal(scatter)
    else:
        # Using cividis colormap for better colorblind accessibility
        cbar = fig.colorbar(scatter, ax=ax, label='Spot Weight [MU]',
                            orientation='vertical')

    # Set major grid and ticks for 1 cm spacing
    ax.set_xticks(np.arange(np.floor(min(x)), np.ceil(max(x)) + 1, 1))
    ax.set_yticks(np.arange(np.floor(min(y)), np.ceil(max(y)) + 1, 1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Set minor grid and ticks for 0.2 cm spacing
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.grid(True, which='major', color='gray', linestyle='--', linewidth=0.5)
    ax.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5)

    ax.set_title(
        f"Field {field_index + 1}, Energy Layer {energy_layer_index + 1} - Beam Energy: {energy if energy != -1 else 'N/A'} MeV")
    ax.set_xlabel('X Position [cm]')
    ax.set_ylabel('Y Position [cm]')
    plt.draw()
    return cbar


def create_interactive_plot(maps, max_weight):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    field_index = [0]  # Current field
    energy_layer_index = [0]  # Current energy layer
    cbar = None  # Initial color bar is None

    # Initial plot and create initial color bar
    cbar = plot_map(field_index[0], energy_layer_index[0],
                    maps, ax, cbar, fig, max_weight)

    # Define button click events, including passing max_weight
    def next_field(event):
        nonlocal cbar
        field_index[0] = (field_index[0] + 1) % len(maps)
        # Reset energy layer index when changing fields
        energy_layer_index[0] = 0
        cbar = plot_map(
            field_index[0], energy_layer_index[0], maps, ax, cbar, fig, max_weight)

    def prev_field(event):
        nonlocal cbar
        field_index[0] = (field_index[0] - 1) % len(maps)
        energy_layer_index[0] = 0
        cbar = plot_map(
            field_index[0], energy_layer_index[0], maps, ax, cbar, fig, max_weight)

    def next_layer(event):
        nonlocal cbar
        energy_layer_index[0] = (
            energy_layer_index[0] + 1) % len(maps[field_index[0]])
        cbar = plot_map(
            field_index[0], energy_layer_index[0], maps, ax, cbar, fig, max_weight)

    def prev_layer(event):
        nonlocal cbar
        energy_layer_index[0] = (
            energy_layer_index[0] - 1) % len(maps[field_index[0]])
        cbar = plot_map(
            field_index[0], energy_layer_index[0], maps, ax, cbar, fig, max_weight)

    # Button placement and creation
    ax_field_prev = plt.axes([0.1, 0.05, 0.1, 0.075])
    ax_field_next = plt.axes([0.21, 0.05, 0.1, 0.075])
    ax_layer_prev = plt.axes([0.68, 0.05, 0.1, 0.075])
    ax_layer_next = plt.axes([0.79, 0.05, 0.1, 0.075])

    b_field_next = Button(ax_field_next, 'Next Field')
    b_field_prev = Button(ax_field_prev, 'Previous Field')
    b_layer_next = Button(ax_layer_next, 'Next Layer')
    b_layer_prev = Button(ax_layer_prev, 'Previous Layer')

    b_field_next.on_clicked(next_field)
    b_field_prev.on_clicked(prev_field)
    b_layer_next.on_clicked(next_layer)
    b_layer_prev.on_clicked(prev_layer)

    plt.show()


def find_global_max_weight(maps):
    """Find the maximum weight across all maps."""
    max_weight = 0
    for field_maps in maps:
        for _, weights, _ in field_maps:
            max_weight = max(max_weight, max(weights))
    return max_weight


def main(args=None):
    """Main routine with option parsing."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='Plot DICOM proton therapy treatment plans.')
    parser.add_argument('inputfile', help='input filename', type=str)
    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help='Increase output verbosity.')
    parsed_args = parser.parse_args(args)

    logging.basicConfig(
        level=logging.INFO if parsed_args.verbosity else logging.WARNING)

    dicom_object = pydicom.dcmread(parsed_args.inputfile)

    maps = []
    for ibs_index in range(len(dicom_object.IonBeamSequence)):
        field_maps = []
        ibs = dicom_object.IonBeamSequence[ibs_index]
        for cps_index in range(len(ibs.IonControlPointSequence)):
            map_data = get_spot_map(dicom_object, ibs_index, cps_index)
            field_maps.append(map_data)
        maps.append(field_maps)

    max_weight = find_global_max_weight(maps)
    create_interactive_plot(maps, max_weight)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
