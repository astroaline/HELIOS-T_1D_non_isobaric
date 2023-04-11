from input import *
import load_files
import numpy as np
import struct
import pdb
from scipy.interpolate import RegularGridInterpolator, interp1d


def load_opacity(temperature, pressure, molecule):
    # res = 0.01   # resolution for the opacities
    step_size = int(res/0.01)

    wavenumber_min = int(1e4/wavelength_bins[-1])
    wavenumber_max = int(1e4/wavelength_bins[0])

    index_min = int((wavenumber_min)/res)
    index_max = int((wavenumber_max)/res)

    if res == 2:
        index_max = index_max - 1

    temp_str = str(temperature).zfill(5)   # temperature as in opacity filename
    pressure_load = int(np.log10(pressure) * 100)

    if pressure_load < 0:
        pressure_str = 'n' + str(abs(pressure_load)).rjust(3, '0')   # pressure as in opacity filename
    else:
        pressure_str = 'p' + str(abs(pressure_load)).rjust(3, '0')

    wavenumber_dict = {'1H2-16O__POKAZATEL_e2b': '42000', '14N-1H3__CoYuTe_e2b': '20000', '12C-1H4__YT10to10_e2b': '13000', '12C-16O2__CDSD_4000_e2b': '09000', '12C-16O__Li2015_e2b': '22000'}

    filename = molecule + '/Out_00000_' + wavenumber_dict[molecule] + '_' + temp_str + '_' + pressure_str + '.bin'

    data = []
    with open(opacity_path + filename, "rb") as f:
        byte = f.read(4)
        while byte:
            data.extend(struct.unpack("f", byte))
            byte = f.read(4)

    x_full = np.r_[0:42000:0.01]

    x_full = x_full[index_min * step_size:index_max * step_size:step_size]
    data = np.array(data[index_min * step_size:index_max * step_size:step_size])

    if len(data) < len(x_full):
        print('padding opacities...')
        data = np.pad(data, (0, len(x_full) - len(data)), mode='constant', constant_values=1e-6)

    return data, x_full



def load_cia(x_full):
    sigma_h2h2 = load_files.load_sigma('H2', 'H2', x_full)
    sigma_h2he = load_files.load_sigma('H2', 'He', x_full)

    sigma_cia = sigma_h2h2 + (solar_he/solar_h2) * sigma_h2he

    return sigma_h2h2, sigma_h2he



def interpolate_opacity(my_pressure, pressure_arr, x_full, opacity):
    fn = RegularGridInterpolator((pressure_arr, x_full), opacity, bounds_error=False, fill_value=None)
    pt = (my_pressure, x_full)
    y = fn(pt)
    return y



def tau(p0_bar):
    # compute tau for all pressures
    pressure_array_pmin_opacities = pressure_array_opacities[np.where(pressure_array_opacities == pmin)[0][0]:]   # remove everything below pmin (this is in bars)

    pressure_levels_pmin_log = np.linspace(np.log10(pmin), np.log10(p0_bar), num_levels)   # log pressure array with num_levels (bars)
    pressure_levels_pmin = 10**pressure_levels_pmin_log
    p0_cgs = p0_bar * 1e6   # convert to cgs

    integral_dict = {}


    ## define molecular opacities ##
    for molecule in molecules:
        _, x_full = load_opacity(temp_dict[molecule][0], pressure_levels_pmin[0], molecule)   # load one to get x_full

        integral_grid_molecule = np.zeros((len(temp_dict[molecule]), len(pressure_levels_pmin), len(x_full)))

        # we will integrate over pressure, for each temperature, for all wavelengths
        for i, t in enumerate(temperature_array):   # load integrands for all pressures
            opacity_grid = np.zeros((len(pressure_array_pmin_opacities), len(x_full)))   # load opacities for all available pressures

            for j, p in enumerate(pressure_array_pmin_opacities):
                opacity_grid[j], _ = load_opacity(t, p, molecule)

            molecular_pressure_interpolator = interp1d(pressure_array_pmin_opacities, opacity_grid, axis=0, bounds_error=False, fill_value=(opacity_grid[0], opacity_grid[-1]), assume_sorted=True)
            opacity_grid_all_levels = molecular_pressure_interpolator(pressure_levels_pmin)
            integral_grid_molecule[i] = opacity_grid_all_levels

        integral_dict[molecule] = interp1d(temperature_array, integral_grid_molecule, axis=0, bounds_error=False, fill_value=(integral_grid_molecule[0], integral_grid_molecule[-1]), assume_sorted=True)   # save interpolator instead


    ## define cia opacities ##
    sigma_h2h2_full, sigma_h2he_full = load_cia(x_full)
    integral_grid_h2h2 = np.zeros((len(temperature_array_cia), len(pressure_levels_pmin), len(x_full)))
    integral_grid_h2he = np.zeros((len(temperature_array_cia), len(pressure_levels_pmin), len(x_full)))

    for i, t in enumerate(temperature_array_cia):
        opacity_grid_h2h2 = np.zeros((len(pressure_array_pmin_opacities), len(x_full)))
        opacity_grid_h2he = np.zeros((len(pressure_array_pmin_opacities), len(x_full)))

        for j, p in enumerate(pressure_array_pmin_opacities):
            opacity_grid_h2h2[j] = sigma_h2h2_full[i]
            opacity_grid_h2he[j] = sigma_h2he_full[i]

        h2h2_pressure_interpolator = interp1d(pressure_array_pmin_opacities, opacity_grid_h2h2, axis=0, bounds_error=False, fill_value=(opacity_grid_h2h2[0], opacity_grid_h2h2[-1]), assume_sorted=True)
        h2he_pressure_interpolator = interp1d(pressure_array_pmin_opacities, opacity_grid_h2he, axis=0, bounds_error=False, fill_value=(opacity_grid_h2he[0], opacity_grid_h2he[-1]), assume_sorted=True)
        opacity_grid_h2h2_all_levels = h2h2_pressure_interpolator(pressure_levels_pmin)
        opacity_grid_h2he_all_levels = h2he_pressure_interpolator(pressure_levels_pmin)
        integral_grid_h2h2[i] = opacity_grid_h2h2_all_levels
        integral_grid_h2he[i] = opacity_grid_h2he_all_levels

    integral_dict['cia_h2h2'] = interp1d(temperature_array_cia, integral_grid_h2h2, axis=0, bounds_error=False, fill_value=(integral_grid_h2h2[0], integral_grid_h2h2[-1]), assume_sorted=True)
    integral_dict['cia_h2he'] = interp1d(temperature_array_cia, integral_grid_h2he, axis=0, bounds_error=False, fill_value=(integral_grid_h2he[0], integral_grid_h2he[-1]), assume_sorted=True)


    ## define rayleigh opacities (independent of temperature) ##
    sigma_rayleigh = np.array([8.4909e-45 * (x_full ** 4)])
    integral_grid_rayleigh = np.zeros((len(pressure_array_pmin_opacities), len(x_full)))

    for j, p in enumerate(pressure_array_pmin_opacities):
        integral_grid_rayleigh[j] = sigma_rayleigh

    rayleigh_pressure_interpolator = interp1d(pressure_array_pmin_opacities, integral_grid_rayleigh, axis=0, bounds_error=False, fill_value=(integral_grid_rayleigh[0], integral_grid_rayleigh[-1]), assume_sorted=True)

    integral_dict['rayleigh'] = rayleigh_pressure_interpolator(pressure_levels_pmin)


    return integral_dict, x_full
