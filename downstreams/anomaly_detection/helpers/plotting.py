import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

"""
PLOTTING KWARGS
"""
kwargs_dict_bands = {
               "SB":  {"density": True,"histtype": "step", "color":"blue", "label": "SB"},
               
               "SBL":  {"density": True,"histtype": "step", "color":"green","label": "SBL"},
               
               "SBH":  {"density": True, "histtype": "step", "color":"purple","label": "SBH"},
               
                "IBL":  {"density": True,"histtype": "step", "color":"orange","label": "IBL"},
               
               "IBH":  {"density": True, "histtype": "step", "color":"pink","label": "IBH"},
               
              "SR":  {"density": True, "histtype": "step",  "color":"red", "label": "SR"},
               
              "SB_samples":  {"density": True,   "histtype": "stepfilled",   "color":"blue",  "label": "SB samples", "alpha": 0.3},
               
               "SBL_samples":  {"density": True,  "histtype": "stepfilled",   "color":"green",  "label": "SBL samples",   "alpha": 0.3},
               
               "SBH_samples":  {"density": True,   "histtype": "stepfilled",   "color":"purple",    "label": "SBH samples",  "alpha": 0.3},
               
               "IBL_samples":  {"density": True,  "histtype": "stepfilled",   "color":"orange",  "label": "IBL samples",   "alpha": 0.3},
               
               "IBH_samples":  {"density": True,   "histtype": "stepfilled",   "color":"pink",    "label": "IBH samples",  "alpha": 0.3},
               
              "SR_samples":  {"density": True,  "histtype": "stepfilled",   "color":"red",    "label": "SR samples",  "alpha": 0.3}
}

kwargs_dict_dtype = {
                "DATA_nojet":  {"density": True, "histtype": "step", "color":"blue", "label": "DATA_nojet"},
      "DATA_jet":  {"density": True, "histtype": "step", "color":"blue", "label": "DATA_jet"},
    "DATA":  {"density": True, "histtype": "step", "color":"blue", "label": "DATA"},
               
              "SM_SIM":  {"density": True,  "histtype": "step", "color":"red", "label": "CMS SIM"},
            "cmssim":  {"density": True,  "histtype": "step", "color":"red", "label": "CMS SIM"},
    
     "s_inj_data":  {"density": True,  "histtype": "step", "color":"red", "label": "s_inj_data"},

             "BSM_XYY":  {"density": True,  "histtype": "step", "color":"green", "label": "BSM_XYY"},
    "wp_wzpythia_forcms_charge-mz50.0-mw40.0-mwp1000_full":  {"density": True,  "histtype": "step", "color":"green", "label": "BSM_XYY"},

             "BSM_HAA":  {"density": True,  "histtype": "step", "color":"orange", "label": "BSM_HAA"},
              }



feature_labels = {
                  "muon_pt": "$\mu$ $p_T$",
                  "amuon_pt": "$\overline{\mu}$ $p_T$",
                  "mu0_pt": "$\mu_0$ $p_T$",
                  "mu1_pt": "$\mu_1$ $p_T$",
                  "muon_eta": "$\mu$ $\eta$",
                  "amuon_eta": "$\overline{\mu}$ $\eta$",
                  "mu0_eta": "$\mu_0$ $\eta$",
                  "mu1_eta": "$\mu_1$ $\eta$",
                  "mu0_phi": "$\mu_0$ $\phi$",
                  "mu1_phi": "$\mu_1$ $\phi$",
                  "mu0_ip3d": "$\mu_0$ IP3D",
                  "mu1_ip3d": "$\mu_1$ IP3D",
                  "mu0_jetiso": "$\mu_0$ jetISO",
                  "mu1_jetiso": "$\mu_1$ jetISO",
                  "mu0_iso03": "$\mu_0$ isoR03",
                  "mu1_iso03": "$\mu_1$ isoR03",
                  "mu0_iso04": "$\mu_0$ isoR04",
                  "mu1_iso04": "$\mu_1$ isoR04",
                  "muon_iso03": "$\mu$ iso R03",
                  "amuon_iso03": "$\overline{\mu}$ iso R03",
                  "muon_iso04": "$\mu$ iso R04",
                  "amuon_iso04": "$\overline{\mu}$ iso R04",
                  "dijet_pt": "Dijet $p_T$",
                  "dijet_eta": "Dijet $\eta$",
                  "dijet_mass": "Dijet $M$",
                  "hardest_jet_pt": "Jet $p_T$",
                  "hardest_jet_eta": "Jet $\eta$",
                  "hardest_jet_phi": "Jet $\phi$",
                  "hardest_jet_mass": "Jet $M$",
                  "jet0_btag": "Hardest jet Jet_btagDeepB",
                  "hardest_jet_btag": "Hardest jet Jet_btagDeepB",
                  "jet1_btag": "Second jet Jet_btagDeepB",
                  "higgs_pt": "$h$ $p_T$",
                  "higgs_eta": "$h$ $\eta$",
                  "higgs_mass": "$h$ $M$",
                  "dimu_pt": "Dimu $p_T$",
                  "dimu_eta": "Dimu $\eta$",
                  "dimu_phi": "Dimu $\phi$",
                  "dimu_mass": "Dimu $M$",
                  "n_electrons": "Num. electrons",
                  "n_muons": "Num. muons",
                  "n_jets": "Num. Jets",
                  "mumu_deltaR": "$\mu\mu$ $\Delta R$",
                  "mumu_deltapT": "$\mu\mu$ $\Delta p_T$",
                  "dimujet_deltaR": "$\mu\mu$ Jet $\Delta R$",
    
        }


n_bins = 60
feature_bins = {
                "mu0_ip3d": np.linspace(0, 0.25, n_bins), 
                "mu1_ip3d": np.linspace(0, 0.25, n_bins), 
                "mu0_jetiso":np.linspace(0, 8, n_bins), 
                "mu1_jetiso":np.linspace(0, 8, n_bins), 
                "mu0_pt": np.linspace(0, 50, n_bins), 
                "mu1_pt":np.linspace(0, 50, n_bins), 
                "muon_eta": np.linspace(-3, 3, n_bins), 
                "amuon_eta": np.linspace(-3, 3, n_bins), 
                "mu0_eta": np.linspace(-3, 3, n_bins), 
                "mu1_eta":np.linspace(-3, 3, n_bins), 
                "mu0_phi":np.linspace(-3.2, 3.2, n_bins), 
                "mu1_phi":np.linspace(-3.2, 3.2, n_bins), 
                "mu0_iso03": np.linspace(0, 1, n_bins),
                "mu1_iso03": np.linspace(0, 1, n_bins),
                "mu0_iso04": np.linspace(0, 1, n_bins),
                "mu1_iso04":np.linspace(0, 1, n_bins),
                "dijet_pt": np.linspace(0, 300, n_bins), 
                "dijet_eta": np.linspace(-6, 6, n_bins), 
                "dijet_mass": np.linspace(0, 100, n_bins), 
                "hardest_jet_pt": np.linspace(0, 300, n_bins), 
                "hardest_jet_eta": np.linspace(-6, 6, n_bins), 
                "hardest_jet_phi": np.linspace(-3.2, 3.2, n_bins), 
                "hardest_jet_mass": np.linspace(0, 300, n_bins), 
                "jet0_btag": np.linspace(0, 1, n_bins),
                "hardest_jet_btag": np.linspace(0, 1, n_bins),
                "jet1_btag": np.linspace(0, 1, n_bins), 
                "higgs_pt": np.linspace(0, 300, n_bins), 
                "higgs_eta": np.linspace(-6, 6, n_bins), 
                "higgs_mass": np.linspace(0, 300, n_bins),
                "dimu_pt": np.linspace(0, 150, n_bins), 
                "dimu_eta": np.linspace(-6, 6, n_bins), 
                "dimu_phi":np.linspace(-3.2, 3.2, n_bins), 
                "dimu_mass": np.linspace(0, 120, n_bins),
                "n_electrons": np.linspace(0, 10, 11),
                "n_muons":np.linspace(0, 10, 11),
                "n_jets":np.linspace(0, 10, 11),
                "mumu_deltaR":np.linspace(0, 2, n_bins),
                "mumu_deltapT":np.linspace(0, 100, n_bins),
                "dimujet_deltaR":np.linspace(0, 2, n_bins),
      }



"""
FUNCTIONS
"""

def hist_all_features_dict(data_dicts, data_labels, feature_set, kwargs_dict, scaled_features=False, plot_bound=3, image_path=None, yscale_log=False, nice_labels=True):
    
        
    scaled_feature_bins = [np.linspace(-plot_bound, plot_bound, n_bins) for i in range(len(feature_set))]   
        
    if image_path:
        p = PdfPages(f"{image_path}.pdf")

    for i, feat in enumerate(feature_set):
        fig = plt.figure(figsize = (5, 3))
        

        for j, data_dict in enumerate(data_dicts):
            if scaled_features:
                plt.hist(data_dict[feat], bins = scaled_feature_bins[i], **kwargs_dict[data_labels[j]])
            else:
                plt.hist(data_dict[feat], bins = feature_bins[feat], **kwargs_dict[data_labels[j]])
                
        if yscale_log:
            plt.yscale("log")
        
        if nice_labels:
            plt.xlabel(feature_labels[feat])
        plt.legend()
        plt.ylabel("Density")
        plt.tight_layout()
        if image_path:
            fig.savefig(p, format='pdf') 
        plt.show()
        
    if image_path: 
        p.close()
        
        
        
def hist_all_features_array(samples, labels, feature_set, plot_bound=3, yscale_log=False):
    scaled_feature_bins = [np.linspace(-plot_bound, plot_bound, 40) for i in range(len(feature_set))]   
    
    n_features = len(feature_set)
    fig, ax = plt.subplots(1, n_features, figsize = (3*n_features, 3))
        

    for i, feat in enumerate(feature_set):
        for j, samp in enumerate(samples):
            ax[i].hist(samp[:,i], bins = scaled_feature_bins[i], histtype = "step", density = True, label = labels[j])
         
        if yscale_log:
            ax[i].set_yscale("log")
        ax[i].set_xlabel(feat)
        ax[i].set_yticks([])
    plt.legend(loc = (0, 1))
    ax[0].set_ylabel("Density")
    plt.subplots_adjust(wspace=0)
    plt.show()
  



# ########## Copied From https://github.com/rikab/rikabplotlib/blob/main/src/rikabplotlib/plot_utils.py ##########

# Constants
DPI = 72
FULL_WIDTH_PX = 510
COLUMN_WIDTH_PX = 245

FULL_WIDTH_INCHES = FULL_WIDTH_PX / DPI
COLUMN_WIDTH_INCHES = COLUMN_WIDTH_PX / DPI

GOLDEN_RATIO = 1.618

def newplot(scale = None, subplot_array = None, width = None, height = None, aspect_ratio = 1,  golden_ratio = False, stamp = None, stamp_kwargs = None, use_tex = True, **kwargs):


    # Determine plot aspect ratio
    if golden_ratio:
        aspect_ratio = GOLDEN_RATIO

    # Determine plot size if not directly set
    if scale is None:
        plot_scale = "full"
    if scale == "full":
        fig_width = FULL_WIDTH_INCHES / aspect_ratio
        fig_height = FULL_WIDTH_INCHES 

        if use_tex:
            plt.style.use('/global/cfs/cdirs/m3246/rikab/dimuonAD/helpers/style_full.mplstyle')

        else:
            plt.style.use('/global/cfs/cdirs/m3246/rikab/dimuonAD/helpers/style_full_notex.mplstyle')

    elif scale == "column":
        fig_width = COLUMN_WIDTH_INCHES / aspect_ratio
        fig_height = COLUMN_WIDTH_INCHES 
        plt.style.use('/global/cfs/cdirs/m3246/rikab/dimuonAD/helpers/style_column.mplstyle')
    else:
        raise ValueError("Invalid scale argument. Must be 'full' or 'column'.")


    if width is not None:
        fig_width = width
    if height is not None:
        fig_height = height

    if subplot_array is not None:
        fig, ax = plt.subplots(subplot_array[0], subplot_array[1], figsize=(fig_width, fig_height), **kwargs)
        stamp_kwargs_default = {"style" : 'italic', "horizontalalignment" : 'right', "verticalalignment" : 'bottom', "transform" : ax[0].transAxes}

    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), **kwargs)
        stamp_kwargs_default = {"style" : 'italic', "horizontalalignment" : 'right', "verticalalignment" : 'bottom', "transform" : ax.transAxes}


    # Plot title
    if stamp_kwargs is not None:
        stamp_kwargs_default.update(stamp_kwargs)

    if stamp is not None:
        # Text in the top right corner, right aligned:
        plt.text(1, 1, stamp, **stamp_kwargs_default)



    return fig, ax


def add_whitespace(ax = None, upper_fraction = 1.333, lower_fraction = 1):

    # handle defualt axis
    if ax is None:
        ax = plt.gca()

    # check if log scale
    scale_str = ax.get_yaxis().get_scale()

    bottom, top = ax.get_ylim()

    if scale_str == "log":
        upper_fraction = np.power(10, upper_fraction - 1)
        lower_fraction = np.power(10, lower_fraction - 1)
    
    ax.set_ylim([bottom / lower_fraction, top * upper_fraction])



# function to add a stamp to figures
def stamp(left_x, top_y,
          ax=None,
          delta_y=0.06,
          textops_update=None,
          boldfirst = True,
          **kwargs):
    
     # handle defualt axis
    if ax is None:
        ax = plt.gca()
    
    # text options
    textops = {'horizontalalignment': 'left',
               'verticalalignment': 'center',
               'transform': ax.transAxes}
    if isinstance(textops_update, dict):
        textops.update(textops_update)
    
    # add text line by line
    for i in range(len(kwargs)):
        y = top_y - i*delta_y
        t = kwargs.get('line_' + str(i))


        if t is not None:
            if boldfirst and i == 0:
                ax.text(left_x, y, r"$\textbf{%s}$" % t, weight='bold', **textops)
            else:
                ax.text(left_x, y, t, **textops)



def plot_event(ax, event, R, filename=None, color="red", title="", show=True):


    pts, ys, phis =event[:,0], event[:, 1], event[:, 2]
    ax.scatter(ys, phis, marker='o', s=2 * pts * 500/np.sum(pts), color=color, lw=0, zorder=10, label="Event")

    # Legend
    # legend = plt.legend(loc=(0.1, 1.0), frameon=False, ncol=3, handletextpad=0)
    # legend.legendHandles[0]._sizes = [150]

    # plot settings
    plt.xlim(-R, R)
    plt.ylim(-R, R)
    plt.xlabel('Rapidity')
    plt.ylabel('Azimuthal Angle')
    plt.title(title)
    plt.xticks(np.linspace(-R, R, 5))
    plt.yticks(np.linspace(-R, R, 5))

    ax.set_aspect('equal')
    if filename:
        plt.savefig(filename)
        plt.show()
        plt.close()
        return ax
    elif show:
        plt.show()
        return ax
    else:
        return ax
    


    # Function to take a list of points and create a histogram of points with sqrt(N) errors, normalized to unit area
def hist_with_errors(ax, points, bins, range, weights = None, show_zero = False, show_errors = True, label = None, **kwargs):

    if weights is None:
        weights = np.ones_like(points)

    hist, bin_edges = np.histogram(points, bins = bins, range = range, weights = weights)
    errs2 = np.histogram(points, bins = bins, range = range, weights = weights**2)[0]

    # Check if density is a keyword argument
    density = kwargs.pop("density", False)

    if density:
        bin_widths = (bin_edges[1:] - bin_edges[:-1])
        errs2 = errs2 / (np.sum(hist * bin_widths))
        hist = hist / np.sum(hist * bin_widths)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = (bin_edges[1:] - bin_edges[:-1])

    if show_errors:
        ax.errorbar(bin_centers[hist > 0], hist[hist > 0], np.sqrt(errs2[hist > 0]), xerr = bin_widths[hist > 0] / 2, fmt = "o", label = label, **kwargs)
    else:
        ax.scatter(bin_centers[hist > 0], hist[hist > 0], label = label, **kwargs)


def hist_with_outline(ax, points, bins, range, weights = None, color = "purple", alpha_1 = 0.25, alpha_2 = 0.75, label = None,  **kwargs):
    
    if weights is None:
        weights = np.ones_like(points)

    ax.hist(points, bins = bins, range = range, weights = weights, color = color, alpha = alpha_1, histtype='stepfilled', **kwargs)
    ax.hist(points, bins = bins, range = range, weights = weights, color = color, alpha = alpha_2, histtype='step', label = label, **kwargs)


    # # # Dummy plot for legend
    # if label is not None:

    #     edgecolor = mpl.colors.colorConverter.to_rgba(color, alpha=alpha_2)
    #     ax.hist(points, bins = bins, range = range, weights = weights * -1, color = color, alpha = alpha_1, lw = lw*2, label = label, edgecolor = edgecolor, **kwargs)




def function_with_band(ax, f, range, params, pcov = None, color = "purple", alpha_line = 0.75, alpha_band = 0.25, lw = 3,  **kwargs):

    x = np.linspace(range[0], range[1], 1000)

    if pcov is not None:

        # Vary the parameters within their errors
        n = 1000
        temp_params = np.random.multivariate_normal(params, pcov, n)
        y = np.array([f(x, *p) for p in temp_params])

        # Plot the band

        y_mean = np.mean(y, axis = 0)
        y_std = np.std(y, axis = 0) 

        ax.fill_between(x, y_mean - y_std, y_mean + y_std, color = color, alpha = alpha_band, **kwargs)


    y = f(x, *params)
    ax.plot(x, y, color = color, alpha = alpha_line, lw = lw, **kwargs)


def read_feature(x, event_info, feature):
    index = list(event_info['INPUTS']['SEQUENTIAL']['Source']).index(feature)
    normal = event_info['INPUTS']['SEQUENTIAL']['Source'][feature]
    input_x = x[..., index]
    if "log" in normal.lower():
        input_x = np.expm1(input_x)
    return input_x

def read_cond(x, event_info, cond):
   index = list(event_info['INPUTS']["GLOBAL"]["Conditions"]).index(cond)
   normal = event_info['INPUTS']["GLOBAL"]["Conditions"][cond]
   input_x = x[..., index]
   if "log" in normal.lower():
        input_x = np.expm1(input_x)
   return input_x
