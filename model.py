from input import *
from load_files import *
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pdb


class Model:

    def __init__(self, len_x, x_full, bin_indices, parameter_dict, integral_dict):
        self.len_x = len_x
        self.x_full = x_full
        self.bin_indices = bin_indices
        self.parameter_dict = parameter_dict
        # self.opacity_grid = opacity_grid
        self.integral_dict = integral_dict


    def molecular_opacity(self, my_temp, opacity_table):
        fn = RegularGridInterpolator((self.x_full, temperature_array), opacity_table)
        pt = (self.x_full, my_temp)
        y = fn(pt)
        return y


    def cia_cross_section(self, my_temp, cia_table):
        fn = RegularGridInterpolator((temperature_array_cia, self.x_full), cia_table)
        pt = (my_temp, self.x_full)
        y = fn(pt)
        return y


    # def integral_grid_interpolate(self, my_temp, temp_arr, pressure_arr, integral_grid):
    #
    #     # pdb.set_trace()
    #
    #     fn = RegularGridInterpolator((temp_arr, pressure_arr, self.x_full), integral_grid) # Interpolate on 3D grid
    #     pts = np.meshgrid(np.array([my_temp]), pressure_arr, self.x_full)
    #     flat_pts = np.array([m.flatten() for m in pts])     # Some complicated fix for 3D
    #     out_array = fn(flat_pts.T)
    #     # y = out_array.reshape(*pts[0].shape)
    #     y = out_array.reshape(len(pressure_arr), len(self.x_full))
    #
    #
    #     return y


    ## calculate transit depth ##
    def transit_depth(self):
        my_temp = self.parameter_dict["T"]
        R0 = self.parameter_dict["R0"] * rjup
        P0 = 10 ** self.parameter_dict["log_P0"]
        Rstar = self.parameter_dict["Rstar"] * rsun
        G = self.parameter_dict["G"]

        p0_bar = P0
        p0_cgs = P0 * 1e6

        pressure_array_pmin = 10 ** np.linspace(np.log10(pmin), np.log10(p0_bar), num_levels) * 1e6
        pressure_levels = pressure_array_pmin

        mass_fraction = []
        molecular_mass = []
        integral_molecules = []

        for molecule in molecules:
            abundance_name = molecular_abundance_dict[molecule]
            mass_fraction.append([10**self.parameter_dict[abundance_name]])
            molecular_mass.append([molecular_mass_dict[molecule]])
            abundance_name = molecular_abundance_dict[molecule]
            integral_mol = self.integral_dict[molecule](my_temp) * (10 ** self.parameter_dict[abundance_name])
            integral_molecules.append(integral_mol * molecular_mass_dict[molecule])

        integral_molecules_sum = np.sum(np.array(integral_molecules), axis=0)

        mass_fraction = np.array(mass_fraction)
        molecular_mass = np.array(molecular_mass)

        background = 1-np.sum(mass_fraction)
        xh2 = background * solar_h2/(solar_h2+solar_he) # Fraction in H2
        xhe = background * solar_he/(solar_h2+solar_he) # Fraction in He

        # xh2 = (1 - np.sum(mass_fraction))/1.1   # calculate abundance of H2 (above is more accurate)

        try:
            m
        except:   # set mean molecular mass if not given in input
            m = 2.0*xh2*amu + 4.0*xhe*amu + np.sum(mass_fraction*molecular_mass)


        if self.parameter_dict["log_P_cloudtop"] == "Off":   # cloud free
            cloud_tau = np.zeros((len(pressure_levels), len(self.x_full)))

        elif self.parameter_dict["Q0"] == "Off":   # grey cloud
            cloudtop_pressure = (10**self.parameter_dict["log_P_cloudtop"])*1e6
            cloud_tau = np.zeros((len(pressure_levels), len(self.x_full)))
            cloud_tau[pressure_levels > cloudtop_pressure,:] = np.inf

        else:   # non-grey cloud
            cloudtop_pressure = (10**self.parameter_dict["log_P_cloudtop"])*1e6
            bc = 10**self.parameter_dict["log_cloud_depth"]
            cloudbottom_pressure = bc*cloudtop_pressure

            Q0 = self.parameter_dict["Q0"]
            a = self.parameter_dict["a"]
            rc = 10**self.parameter_dict["log_r_c"]
            tau_ref = 10**self.parameter_dict["log_tau_ref"]
            x_lambda = 2 * np.pi * rc * self.x_full
            x_lambda_ref = 2 * np.pi * rc * 1e4
            tau_nongrey = tau_ref * (Q0 * x_lambda_ref ** (-a) + x_lambda_ref ** 0.2) / (Q0 * x_lambda ** (-a) + x_lambda ** 0.2)

            cloud_tau = np.zeros((len(pressure_levels), len(self.x_full)))
            cloud_tau[(pressure_levels > cloudtop_pressure) & (pressure_levels < cloudbottom_pressure),:] = tau_nongrey   # array of len(pressure_levels)


        scale_height = kboltz * my_temp / m / G

        integral_cia_h2h2 = self.integral_dict['cia_h2h2'](my_temp)
        integral_cia_h2he = self.integral_dict['cia_h2he'](my_temp)

        integral_grid_rayleigh = self.integral_dict['rayleigh'] * xh2

        tau_values = np.zeros((len(pressure_levels), len(self.x_full)))

        for i, p in enumerate(pressure_levels):
            ntot = p / kboltz / my_temp   # p already in cgs
            integral_grid_cia = xh2 * ntot * (xh2 * integral_cia_h2h2 + 0.1 * xh2 * integral_cia_h2he)

            r = scale_height * np.log(p0_cgs / p)
            factor = p * np.sqrt(2 * np.pi * scale_height * (R0 + r)) / kboltz / my_temp   # tau factor missing sigma

            sigma_values = integral_molecules_sum[i] + integral_grid_cia[i] + integral_grid_rayleigh[i]   # array of (len(pressure_levels), len(x_full))
            tau_values[i] = sigma_values * factor   # array of (len(pressure_levels), len(x_full))


        tau_values_int = tau_values + cloud_tau

        h_integrand1 = (R0 + scale_height * np.log(p0_cgs / pressure_levels)) / pressure_levels   # array of len(pressure_levels)
        h_integrand2 = 1 - np.exp(-tau_values_int)   # array of (len(pressure_levels), len(x_full))
        h_integrand = (h_integrand2.T * h_integrand1).T   # vectorised h2*h1, gives array of (len(pressure_levels), len(x_full))
        h_integral_values = np.trapz(h_integrand, pressure_levels, axis=0)   # do the integral over p, gives array of len(x_full)

        h_values = (scale_height / R0) * h_integral_values
        r = R0 + h_values

        result = 100.0 * (r / Rstar) ** 2  # return percentage transit depth

        return result




    def binned_model(self):
    ## calculates average transit depth in given bins ##

        if self.parameter_dict['line'] == 'Off':
            y_full = self.transit_depth()
            y_mean = np.zeros(self.len_x)
            for i in range(self.len_x):
                j = int(self.bin_indices[i])
                k = int(self.bin_indices[i + 1])
                y_mean[i] = np.mean(y_full[k:j])  # bin transit depth      
        else:
            y_mean = np.full(self.len_x, self.parameter_dict['line'])
            
        return y_mean
