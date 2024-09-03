from itertools import permutations
import sys
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
import glob, os
import pandas as pd
from skimage import measure
from skimage.transform import hough_line, hough_line_peaks
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from pyimzml.ImzMLParser import ImzMLParser, getionimage
from skimage.filters import gaussian
from skimage.morphology import binary_dilation, disk
import tqdm
import re

def convert_molecule_name(name):
    # Find the molecular type
    molecular_type_match = re.match(r'([A-Za-z]+)', name)

    if not molecular_type_match or str(name) == 'False':
        return ''
    
    molecular_type = molecular_type_match.group(0)

    # Find all occurrences of 'digit+:digit+'
    parts = re.findall(r'(\d+):(\d+)', name)
    if not parts:
        return False  # Return the original name if no molecular type is found
    
    # Sum the digits for carbons and unsaturations
    total_carbons = sum(int(carbon) for carbon, _ in parts)
    total_unsaturations = sum(int(unsaturation) for _, unsaturation in parts)

    # Recompose the name
    return f'{molecular_type}({total_carbons}:{total_unsaturations})'

def scale(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def contrast(arr, low, high):
    return np.clip(arr, np.percentile(arr, low), np.percentile(arr, high))

def line_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])

    if np.linalg.cond(A) < 1/sys.float_info.epsilon:
        # A is not singular, solve the system
        x, y = np.linalg.solve(A, b)
        return [x[0], y[0]]
    else:
        # Lines are parallel or almost parallel
        return None

def rotate(centroid_coords, rotation):
    xe, ye = centroid_coords[:, 0], centroid_coords[:, 1]

    theta = rotation / 180. * np.pi
    x_spots = np.mean(xe) + np.cos(theta) * (xe - np.mean(xe)) \
                - np.sin(theta) * (ye - np.mean(ye))
    y_spots = np.mean(ye) + np.sin(theta) * (xe - np.mean(xe)) \
                + np.cos(theta) * (ye - np.mean(ye))

    return np.array([x_spots, y_spots]).T

def estimateAngle(centroid_coords, shape, Figures=False):
    """Estimate the relative angle between the ablation marks and the X axis.
    TODO: estimate in both X and Y and get average estimate. It might be more accurate
    Args:
        xe (array): X coordinates of the ablation marks (1D).
        ye (array): Y coordinates of the ablation marks (1D).
        shape (array): number of rows and columns of the MALDI acquisition (1D).
        MFA (str): path to Main Folder Analysis.
        Figures (bool): whether or not plot the results and save in analysis folder.

    Returns:
        rotation_deg (float): alignment rotation angle in degree to the X axis.

    """
    xe, ye = centroid_coords[:, 0], centroid_coords[:, 1]
    counts_x = []
    counts_y = []
    angle = []
    for i in np.linspace(-10, 10, 5000):
        rotation = i
        theta = rotation / 180. * np.pi
        x_spots = np.mean(xe) + np.cos(theta) * (xe - np.mean(xe)) \
                    - np.sin(theta) * (ye - np.mean(ye))
        y_spots = np.mean(ye) + np.sin(theta) * (xe - np.mean(xe)) \
                    + np.cos(theta) * (ye - np.mean(ye))
        a_x = np.histogram(x_spots, int(shape[0]) * 100)
        a_y = np.histogram(y_spots, int(shape[0]) * 100)
        count_x = np.asarray(a_x[0][a_x[0] > 0]).shape[0]
        count_y = np.asarray(a_y[0][a_y[0] > 0]).shape[0]
        # print count, theta
        angle.append(rotation)
        counts_x.append(count_x)
        counts_y.append(count_y)
        # print '{}/{} deg'.format(i, 15)
    rotation_deg_x = angle[np.where(counts_x == np.min(counts_x))[0][0]]
    # rotation_deg_y = angle[np.where(counts_y == np.min(counts_y))[0][0]]
    # rotation_deg = np.mean([rotation_deg_x,rotation_deg_y])
    # print('Rough Rotation estimation is {} degree'.format(rotation_deg_x))
    if Figures == True:
        plt.figure()
        plt.plot(angle, counts_x, 'k')
        plt.plot(rotation_deg_x, np.min(counts_x), 'ro', markersize=15)
        plt.legend()
        plt.xlabel('Rotation angles (rad)', fontsize=15)
        plt.ylabel('Number of non-zero bins \nof data 1D projection', fontsize=15)
        # plt.title('Angle Estimation: angle=%.3f degree' % (rotation_deg_x), fontsize=10)
        # plt.close('all')
    return rotation_deg_x

def sample_one_line_per_cluster(data, n_clusters=70):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)

    # Get centroids of the clusters
    centroids = kmeans.cluster_centers_

    # Sort the centroids (example: based on the first coordinate)
    sorted_cluster_indices = np.argsort(centroids[:, 0])

    sampled_lines = []
    for idx in sorted_cluster_indices:
        lines_in_cluster = data[labels == idx]
        if len(lines_in_cluster) > 0:
            sampled_lines.append(lines_in_cluster[0])  # sample the first line in each cluster
    return sampled_lines

def apply_transformations(matrix, transformations):
    
    tf = np.copy

    for transform in transformations:
        if transform == 'flip_lr':
            tf = np.fliplr
        elif transform == 'flip_ud':
            tf = np.flipud
        elif transform == 'transpose':
            tf = np.transpose
        
    return tf(matrix), tf
    
def find_transformation(source_matrix, target_matrix, figure=False):
    # Function to compute spatial correlation
    def compute_correlation(source, target):
        return np.corrcoef(source.flatten(), target.flatten())[0, 1]

    # Generate all unique and non-redundant combinations of transformations
    transformations = ['flip_lr', 'flip_ud', 'rotate90', 'rotate180', 'rotate270', 'transpose']
    all_combinations = []
    for r in range(len(transformations) + 1):
        for subset in permutations(transformations, r):
            if subset not in all_combinations:
                all_combinations.append(subset)

    # Main comparison logic
    results = []
    for combo in all_combinations:
        transformed_matrix, tf = apply_transformations(source_matrix, combo)
        correlation = compute_correlation(transformed_matrix, target_matrix)
        results.append((combo, correlation, len(combo)))

    # Sort results by absolute correlation, then by length of transformation sequence
    results.sort(key=lambda x: (-abs(x[1]), x[2]))

    # Extract the best transformation
    best_transform, best_correlation, _ = results[0]
    transformed_matrix, tf = apply_transformations(source_matrix, best_transform)

    # Print the best transformation and its correlation
    print("Best Transformation:", ', '.join(best_transform) if best_transform else 'No Transformation')
    print("Correlation:", best_correlation)

    # Plotting the source image and the transformed target image
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(target_matrix, cmap='gray')
    plt.title('TIC')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(transformed_matrix, cmap='gray')
    plt.title('AM-cell overlap')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    rgb = np.zeros((transformed_matrix.shape[0], transformed_matrix.shape[1], 3))
    rgb[..., 0] = transformed_matrix
    rgb[..., 1] = target_matrix
    plt.imshow(rgb)
    plt.title('Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return transformed_matrix, tf, best_correlation

def filter_rows_by_zero_freq(df, threshold):
    """
    Removes rows from the DataFrame where the frequency of zeros is above the given threshold.

    :param df: DataFrame to filter.
    :param threshold: Threshold for zero frequency (between 0 and 1).
    :return: Filtered DataFrame.
    """
    zero_freq = (df == 0).sum(axis=1) / df.shape[1]
    filtered_df = df[zero_freq <= threshold]
    
    return filtered_df

def plot_zero_frequency_histogram(df):
    """
    Plots a histogram of the frequency of zeros in each row of the DataFrame.

    :param df: DataFrame to analyze.
    """
    zero_freq = (df == 0).sum(axis=1) / df.shape[1]
    plt.hist(zero_freq, bins=100)
    plt.title('Histogram of Zero Frequencies in Rows')
    plt.xlabel('Frequency of Zeros')
    plt.ylabel('Number of Rows')
    plt.show()

def zscore_columns_with_nonzero_median(df):
    """
    Applies z-score normalization to each column of the DataFrame based on the median and standard deviation
    of the non-zero values in each column.

    :param df: DataFrame to normalize.
    :return: DataFrame with normalized columns.
    """
    zscored_df = pd.DataFrame()
    for col in df.columns:
        non_zero_values = df[col][df[col] > 0]
        median = np.median(non_zero_values)
        std_dev = np.std(non_zero_values)

        if std_dev > 0:
            zscored_df[col] = (df[col] - median) / std_dev
        else:
            # In case of a standard deviation of zero, return the original values
            # This might need handling based on the specific requirements of the analysis.
            zscored_df[col] = df[col]

    return zscored_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: make_training_patches.py <dataset_name>")
        sys.exit(1)
    
    sample = sys.argv[1]

    root = r'MetaLens\data\raw_data'
    shape = (70, 70)
    imzML_root = root
    window_size = 64 # n pixels around the AM centroid (training patch size is window_size*2xwindow_size*2 pixels)
    save_folder = r'MetaLens\data\training_data/'

    imzml_path = glob.glob(fr"{imzML_root}\{sample}.imzML")[0]
    cells = tif.imread(fr"{root}\{sample}_cells.tif")
    bf = scale(tif.imread(fr"{root}\{sample}_ablation_marks_tf.tif"))
    am_probability = tif.imread(fr"{root}\{sample}_ablation_marks_tf_pred.tif")
    cell_mask = tif.imread(fr"{root}\{sample}_cells_mask.tif")

    # correct for microscope objective shift
    cells[..., 2] = np.roll(cells[..., 2], shift=3, axis=0)  # Shift down by 3 pixels
    cells[:3, :, 2] = np.median(cells[..., 2])  # Fill the top 3 rows with the median value

    input_channels = [0, 1, 2]

    am_area_threshold = [0, 400]

    # viewer = napari.Viewer()
    # viewer.add_image(cells[..., 0])
    # viewer.add_image(cells[..., 1])
    # viewer.add_image(cells[..., 2], colormap='blue')

    plt.style.use('default')
    # Presentation_images
    plt.figure(figsize=(20, 20))
    ax = plt.subplot(221)
    plt.imshow(contrast(cells[..., 1], 0.5, 99))
    plt.subplot(222, sharex=ax, sharey=ax)
    plt.imshow(bf, cmap='gray')
    plt.subplot(223, sharex=ax, sharey=ax)
    plt.imshow(am_probability)
    plt.subplot(224, sharex=ax, sharey=ax)
    plt.imshow(contrast(cells[..., 0], 0.5, 99.5), cmap='gray')

    binary_image = am_probability > 0.5
    labels = measure.label(binary_image.astype(int))
    blobs = measure.regionprops(labels)
    centroid_coords = np.array([blob.centroid for blob in blobs])
    xe, ye = centroid_coords[:, 0], centroid_coords[:, 1]
    pix_size = 1

    rotation_esti = estimateAngle(centroid_coords, shape, Figures=True)

    rotation_esti_rad_0 = rotation_esti / 180. * np.pi
    rotation_esti_rad_90 = (rotation_esti + 90) / 180. * np.pi

    tested_angles_0 = np.linspace(0.999*rotation_esti_rad_0, 1.001*rotation_esti_rad_0, 500, endpoint=False)
    tested_angles_90 = np.linspace(0.999*rotation_esti_rad_90, 1.001*rotation_esti_rad_90, 500, endpoint=False)

    tested_angles = np.concatenate((tested_angles_0, tested_angles_90))
    h, theta, d = hough_line(binary_image.astype(int), theta=tested_angles)

    # very permissive threshold as the angle restriction is stringent and clustering will be applied later
    peaks = hough_line_peaks(h, theta, d, threshold=0.1*h.max()) 

    mean_angle = np.mean(peaks[1])

    group_1 = np.array([(d, angle) for _, angle, d in zip(*peaks) if angle > mean_angle])
    group_2 = np.array([(d, angle) for _, angle, d in zip(*peaks) if angle < mean_angle])

    # Apply KMeans , sort clusters and get sampled lines for each group
    sampled_lines_group_1 = sample_one_line_per_cluster(data=group_1, n_clusters=shape[0]) # Horizontal lines
    sampled_lines_group_2 = sample_one_line_per_cluster(data=group_2, n_clusters=shape[1]) # Vertical lines

    # Flag double horizontal lines
    if any(np.diff(np.array(sampled_lines_group_1)[:, 0]) < 15):
        print('Double horizontal lines detected')
        double_index = np.where(np.diff(np.array(sampled_lines_group_1)[:, 0]) < 15)[0][0]
        sampled_lines_group_1 = np.delete(sampled_lines_group_1, double_index, axis=0)
        sampled_lines_group_1 = np.vstack([sampled_lines_group_1[0].copy(), sampled_lines_group_1])
        sampled_lines_group_1 = [i for i in sampled_lines_group_1]

    # # Visualization of clustering and sampled lines
    plt.figure()
    plt.subplot(211)
    plt.scatter(group_2[:, 0], group_2[:, 1], 10, 'k')
    plt.scatter(group_1[:, 0], group_1[:, 1], 10, 'k')
    plt.ylabel(r'$a$ , slope parameter', fontsize=30)
    plt.xlabel(r'$b$ , y-intercept parameter', fontsize=30)

    plt.subplot(212)
    plt.scatter(group_2[:, 0], group_2[:, 1], 10, 'k')
    plt.scatter(group_1[:, 0], group_1[:, 1], 10, 'k')
    plt.ylabel(r'$a$ , slope parameter', fontsize=30)
    plt.xlabel(r'$b$ , y-intercept parameter', fontsize=30)

    plt.subplot(211)
    kmeans = KMeans(n_clusters=shape[0])
    labels = kmeans.fit_predict(group_1)
    centroids = kmeans.cluster_centers_

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for lbl in np.unique(labels):
        color = colors[lbl % len(colors)]  # Use modulo to cycle through colors
        plt.scatter(group_1[labels == lbl, 0], group_1[labels == lbl, 1], s=10, alpha=0.5, color=color)
        plt.scatter(centroids[lbl, 0], centroids[lbl, 1], s=100, color=color)

    plt.subplot(212)
    kmeans = KMeans(n_clusters=shape[1])
    labels = kmeans.fit_predict(group_2)
    centroids = kmeans.cluster_centers_

    # Get the color cycle from current axes
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for lbl in np.unique(labels):
        color = colors[lbl % len(colors)]  # Use modulo to cycle through colors
        # Plot each group with its corresponding color
        plt.scatter(group_2[labels == lbl, 0], group_2[labels == lbl, 1], s=10, alpha=0.5, color=color)
        # Plot centroids with the same color as their groups
        plt.scatter(centroids[lbl, 0], centroids[lbl, 1], s=100, color=color)

    intersections = []
    for line1 in sampled_lines_group_1:
        for line2 in sampled_lines_group_2:
            intersec = line_intersection(line1, line2)
            if intersec is not None:
                intersections.append(intersec)

    intersections = np.array(intersections)

    # plt.close('all')
    plt.figure(figsize=(20, 20))
    # ax = plt.subplot(121)
    plt.imshow(bf, cmap='gray')

    # Plotting lines from both groups
    for dist, angle in sampled_lines_group_1 + sampled_lines_group_2:
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), c='k', linewidth=0.5)

    # Plotting intersections
    plt.scatter(intersections[:, 0], intersections[:, 1], 20, c=range(intersections.shape[0]), zorder=10, cmap='jet')
    plt.axis('off')
    plt.tight_layout()

    # Create a KDTree for efficient nearest-neighbor search
    tree = KDTree(np.fliplr(centroid_coords))

    # Filter area based on their distance to their theoretical position 
    # to discard missing ones/dusts/artefacts
    filtered_centroids = []
    distances = []
    for intersec in intersections:
        # Query the KDTree for the nearest centroid
        dist, idx = tree.query(intersec)
        # Retrieve the label of the nearest centroid
        filtered_centroids.append(centroid_coords[idx])
        distances.append(dist)

    threshold = 25
    filtered_centroids = np.array(filtered_centroids)
    distances = np.array(distances)
    am_distance_pass = distances < threshold

    # Filter am based on area to discard merged ones
    am_areas = np.array([blob.area for blob in blobs])

    plt.figure()
    plt.hist(am_areas, 100), plt.yscale('log')
    plt.axvline(x=am_area_threshold[0], color='r', linestyle='--', linewidth=3)
    plt.axvline(x=am_area_threshold[1], color='r', linestyle='--', linewidth=3)
    plt.xlabel('Area of ablation marks', fontsize=20)
    plt.ylabel('Number of ablation marks', fontsize=20)
    plt.tight_layout()

    am_area_pass = []
    for xy in filtered_centroids:
        am_area_pass.append((am_areas[np.max(centroid_coords == xy, axis=1)][0] < am_area_threshold[1]) and \
                            (am_areas[np.max(centroid_coords == xy, axis=1)][0] > am_area_threshold[0]))

    am_filter_pass = am_distance_pass & am_area_pass

    plt.figure(figsize=(15, 15))
    plt.imshow(bf, cmap='gray')
    plt.scatter(filtered_centroids[am_filter_pass, 1], filtered_centroids[am_filter_pass, 0], 100, c=range(filtered_centroids[am_filter_pass].shape[0]), zorder=10, cmap='viridis')
    plt.scatter(intersections[~am_filter_pass, 0], intersections[~am_filter_pass, 1], 50, c='r', zorder=10)

    on_sample_labels, sampling_counts = np.unique(np.array(labels) * np.array(cell_mask > 0), return_counts=True)
    os_matrix = np.zeros((shape[0]* shape[1]))
    sc_matrix = np.zeros((shape[0]* shape[1]))

    for i, (dist, (x, y)) in enumerate(zip(distances, filtered_centroids)):

        label_ind = np.where(centroid_coords == (x, y))[0][0] + 1

        on_sample = label_ind in on_sample_labels
        if on_sample: sampling_count = sampling_counts[np.where(on_sample_labels == label_ind)[0][0]]
        else: sampling_count = 0

        os_matrix[i] = on_sample
        sc_matrix[i] = sampling_count

    os_matrix = os_matrix.reshape(shape)
    sc_matrix = sc_matrix.reshape(shape)

    dsname = imzml_path.split('\\')[-1].split('.')[0]
    p = ImzMLParser(imzml_path)

    tic = np.zeros(shape)
    for idx, (x,y,z) in enumerate(p.coordinates):
        mzs, intensities = p.getspectrum(idx)
        tic[x-1, y-1] = np.sum(intensities)
        
    source_matrix = scale(contrast(sc_matrix, 1, 99))
    target_matrix = scale(contrast(tic, 1, 99))

    transformed_matrix, best_transform, best_correlation = find_transformation(source_matrix, target_matrix, figure=True)

    # intracellular ions with mean log10 values superior to 0.1
    selected_mol =[['TG(50:1)', 855.7412116029999], ['PC(32:0)', 734.5694319549999], ['', 577.519037091], ['PC(34:1)', 760.585082019], ['', 471.02993499499996], ['', 551.5033870269999], ['TG(52:2)', 881.756861667], ['TG(54:3)', 907.7725117309999], ['PC(34:1)', 782.567026267], ['PC(32:0)', 756.5513762029999], ['TG(56:6)', 907.774917419], ['PC(36:2)', 786.6007320829999], ['', 582.2729461229999], ['TG(54:5)', 881.7592673549999], ['LysoPC(16:0)', 518.321710623], ['LysoPC(16:0)', 496.33976637500007], ['PC(34:2)', 758.5694319549999], ['', 496.3397296670001], ['DG(36:2)', 643.527196087], ['PC(36:2)', 808.5826763309999], ['PC(34:2)', 780.5513762029999], ['SM(34:1)', 703.5748516829999], ['SM(34:1)', 725.5567959309999], ['PC(35:4)', 790.535726139], ['LysoPC(18:1)', 544.337360687], ['PC(33:2)', 766.535726139], ['PA(36:2)', 723.493526979], ['DG(34:1)', 617.511546023], ['LysoPC(18:1)', 522.355416439], ['PA(34:1)', 697.4778769149999], ['PC(33:2)', 744.5537818909999], ['PC(36:3)', 806.567026267], ['PC(36:4)', 804.5513762029999], ['', 582.271095655], ['', 349.19105444300004], ['PC(38:4)', 810.6007320829999], ['PC(35:4)', 768.5537818909999], ['', 575.5033870269999], ['PC(35:5)', 788.5200760749999], ['SM(34:0)', 744.554208237], ['DG(32:0)', 591.495895959], ['PC(32:1)', 732.5537818909999], ['MG(18:1)', 379.28188044300003], ['PC(38:4)', 832.5826763309999], ['PA(38:3)', 749.509177043], ['PA(36:3)', 721.4778769149999], ['PC(31:1)', 740.5200760749999], ['LysoPC(18:2)', 542.321710623], ['PC(36:1)', 788.616382147], ['PC(32:1)', 754.535726139], ['', 353.26623037900004], ['LysoPC(18:2)', 520.339766375], ['LysoPC(18:0)', 524.371066503], ['LysoPC(18:0)', 546.3530107509999], ['', 335.013221845], ['PC(36:3)', 784.585082019], ['PC(38:5)', 830.567026267], ['PC(38:5)', 808.585082019], ['PC(36:4)', 782.5694319549999], ['PA(38:4)', 747.493526979], ['PA(38:5)', 745.4778769149999], ['', 335.01409121300003], ['PC(40:7)', 832.585082019], ['', 958.577984875], ['PC(38:6)', 806.5694319549999], ['', 599.500981339], ['PC(33:1)', 768.5513762029999], ['PC(40:8)', 830.5694319549999], ['', 544.996427641], ['', 413.26623037900004], ['', 840.596155877], ['', 307.018307225]]

    intensities_df = pd.DataFrame()
    for condensed_mol_name, mz_value in tqdm.tqdm(selected_mol):
        # Compute the image and its transformations
        image_imzml = np.log10(getionimage(p, mz_value, tol=mz_value*3*1e-6)+1).T
        image_imzml_tic = image_imzml / np.log10(tic+1)

        adjusted_image = contrast(best_transform(image_imzml_tic), 0.5, 99.5).ravel()

        col_name = f'{condensed_mol_name}_{mz_value:.4f}'

        # Check if the column already exists
        if col_name in intensities_df.columns:
            # Sum the values with the existing column
            intensities_df[col_name] += adjusted_image
        else:
            # Create a new column
            intensities_df[col_name] = adjusted_image

    # Mask out other am in the FOV
    am_mask = np.zeros((window_size*2, window_size*2))
    am_mask[window_size, window_size] = 1
    am_dilated = binary_dilation(am_mask, footprint=disk(30))
    am_mask1 = gaussian(am_dilated, sigma=2)

    show_am_viz = False

    from scipy.spatial.distance import cdist
    # from scipy.stats import pearsonr, spearmanr
    from skimage import measure, morphology

    def mask_central_am(crop_am_prob):

        binary_image = crop_am_prob > 0.5

        labeled_image = measure.label(binary_image)
        
        image_center = np.array([[labeled_image.shape[0] / 2, labeled_image.shape[1] / 2]])
        properties = measure.regionprops(labeled_image)
        centroids = np.array([prop.centroid for prop in properties])
        distances = cdist(centroids, image_center)
        centermost_label = properties[np.argmin(distances)].label
        centermost_mask = labeled_image == centermost_label
        dilated_mask = morphology.dilation(centermost_mask, morphology.disk(3))

        return dilated_mask

    if show_am_viz:
        plt.figure(figsize=(10, 10))
        plt.imshow(contrast(cells[..., 1], 0.5, 99.9), cmap='viridis')
        j = 0 # metabolite index to visualize

    df_oi = intensities_df
    for i in tqdm.tqdm(list(df_oi.index)):

        if am_filter_pass[i] and \
        filtered_centroids[i, 0] > window_size and \
        filtered_centroids[i, 1] > window_size and \
        filtered_centroids[i, 0] < cells.shape[0] - window_size and \
        filtered_centroids[i, 1] < cells.shape[1] - window_size and \
        i > shape[0]*2: # skip first two rows as am are often overlapping

            training_image = [scale(contrast(cells[
                int(filtered_centroids[i, 0])-window_size:int(filtered_centroids[i, 0])+window_size, 
                int(filtered_centroids[i, 1])-window_size:int(filtered_centroids[i, 1])+window_size,
                j], 0.1, 99.9)) for j in input_channels]
            
            crop_am_prob = am_probability[
                int(filtered_centroids[i, 0])-window_size:int(filtered_centroids[i, 0])+window_size, 
                int(filtered_centroids[i, 1])-window_size:int(filtered_centroids[i, 1])+window_size
                ] 
            
            am_mask = mask_central_am(crop_am_prob)

            crop_am_prob = crop_am_prob * am_mask  # mask out neighboring ablation marks

            crop_am_bf = bf[
                int(filtered_centroids[i, 0])-window_size:int(filtered_centroids[i, 0])+window_size, 
                int(filtered_centroids[i, 1])-window_size:int(filtered_centroids[i, 1])+window_size
                ]# * am_mask1 # mask out neighboring ablation marks

            training_image.append(crop_am_bf) # to discard in future versions
            training_image.append(crop_am_prob) # Guides the network to focus on the ablation mark of interest
            # Shows the actual bf image of the ablation marks (more information than the probability map alone)

            training_image = np.dstack(training_image)

            if show_am_viz: plt.scatter(filtered_centroids[i, 1], filtered_centroids[i, 0], s=200, c=df_oi.iloc[i, j], cmap='magma', vmin=df_oi.iloc[:, j].min(), vmax=df_oi.iloc[:, j].max())

            tif.imwrite(save_folder + rf'{sample}_{i}.tif', training_image.astype(np.float32))
            df_oi.loc[i, 'filename'] = f'{sample}_{i}.tif'

    df_oi[df_oi['filename'] != 'nan'].to_csv(save_folder + rf'ion_intensities.csv', index=False)

    plt.close('all')