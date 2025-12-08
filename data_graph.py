import os
import matplotlib.pyplot as plt

# Data
noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

gaussian_overall = [
    3.8715068101882935,
    3.970553827285767,
    4.120978808403015,
    4.266230273246765,
    4.383557772636413,
    4.479494857788086,
    4.548664093017578,
    4.589440703392029,
    4.626200699806214,
    4.656160187721253,
]

gaussian_age = [
    12.0183,
    12.376895,
    12.91586,
    13.46105,
    13.879130000000004,
    14.22606,
    14.480615,
    14.632160000000002,
    14.75875,
    14.85704,
]

gaussian_condition = [
    85.0985,
    83.3005,
    78.7695,
    75.3175,
    69.9265,
    68.044,
    63.74,
    63.1965,
    60.5685,
    58.6115,
]

uniform_overall = [
    3.8300487875938414,
    3.8684197664260864,
    3.9642159700393678,
    4.019692528247833,
    4.112554621696472,
    4.202331447601319,
    4.306445932388305,
    4.388829445838928,
    4.475293469429016,
    4.499164891242981,
]

uniform_age = [
    11.863835,
    12.000745,
    12.362195,
    12.56647,
    12.88712,
    13.202445,
    13.588715,
    13.89818,
    14.22108,
    14.28709,
]

uniform_condition = [
    85.677,
    85.131,
    82.168,
    81.9045,
    78.183,
    76.6315,
    73.2,
    69.0805,
    67.756,
    66.0355,
]

# Output folder
output_folder = "./noise_plots"
os.makedirs(output_folder, exist_ok=True)


# Helper to make & save plot
def save_plot(x, y, title, filename, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    out_path = os.path.join(output_folder, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


saved_files = []

saved_files.append(
    save_plot(
        noise_levels,
        gaussian_overall,
        "Gaussian Noise: Overall Loss",
        "gaussian_overall.png",
        "Noise Level",
        "Avg Overall Loss",
    )
)
saved_files.append(
    save_plot(
        noise_levels,
        gaussian_age,
        "Gaussian Noise: Age Loss (MAE)",
        "gaussian_age.png",
        "Noise Level",
        "Avg Age Loss (MAE)",
    )
)
saved_files.append(
    save_plot(
        noise_levels,
        gaussian_condition,
        "Gaussian Noise: Condition Accuracy",
        "gaussian_condition.png",
        "Noise Level",
        "Condition Accuracy (%)",
    )
)

saved_files.append(
    save_plot(
        noise_levels,
        uniform_overall,
        "Uniform Noise: Overall Loss",
        "uniform_overall.png",
        "Noise Level",
        "Avg Overall Loss",
    )
)
saved_files.append(
    save_plot(
        noise_levels,
        uniform_age,
        "Uniform Noise: Age Loss (MAE)",
        "uniform_age.png",
        "Noise Level",
        "Avg Age Loss (MAE)",
    )
)
saved_files.append(
    save_plot(
        noise_levels,
        uniform_condition,
        "Uniform Noise: Condition Accuracy",
        "uniform_condition.png",
        "Noise Level",
        "Condition Accuracy (%)",
    )
)

saved_files
