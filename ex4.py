# ex4 IMPR, Hebrew U, Avinoam Nukrai

# need to use: feature extraction, feature matching, RANSAC and image warping.



import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree
import sol4_utils




import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter, convolve
from scipy.ndimage import label, center_of_mass, map_coordinates
import shutil
from imageio import imwrite
import sol4_utils

K = 0.04
KERNEL_SIZE = 3
RADIUS = 3


def calc_response_image(x_der_squared_blur, y_der_squared_blur, xy_der_blur):
    """
    This func is responsible for calculating the response image for the given
    variables of the sq ser images
    :param x_der_squared_blur: the squared derivative in the x axis
    :param y_der_squared_blur: the squared derivative in the y axis
    :param xy_der_blur: the squared derivative in the xy axis
    :return: the response image ( = det(M) - K*(trace(M)**2)) M is the
    derivatives matrix
    """
    det_m = x_der_squared_blur * y_der_squared_blur - xy_der_blur * xy_der_blur
    trace_m = y_der_squared_blur + y_der_squared_blur
    return det_m - K * (trace_m * trace_m)


def harris_corner_detector(im):
    """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    # get Ix and Iy derivatives of the image
    filter_vec = np.array([1, 0, -1]).reshape(1, 3)
    x_derivative, y_derivative = convolve(im, filter_vec), convolve(im, filter_vec.transpose())
    # blur the squared der images
    x_der_squared_blur = sol4_utils.blur_spatial(x_derivative * x_derivative, KERNEL_SIZE)
    y_der_squared_blur = sol4_utils.blur_spatial(y_derivative * y_derivative, KERNEL_SIZE)
    xy_der_blur = sol4_utils.blur_spatial(x_derivative * y_derivative, KERNEL_SIZE)
    # calc args for R i.e. get the response image
    response_image = calc_response_image(x_der_squared_blur, y_der_squared_blur, xy_der_blur)
    # extract all the local maximum values in each patch
    local_max_image = non_maximum_suppression(response_image)
    # taking all the values according to a positive treshold
    true_points = np.argwhere(local_max_image)
    return np.flip(true_points, axis=1)  # returning as [column,row]


def calc_pixel_environment(x_coord, y_coord, radius):
    """
    This function is calculating the environment of 2 pixel in a given radius
    :param x_coord: x coordinate
    :param y_coord: y coordinate
    :param radius: the radius of the env
    :return: np.array with all the points of the environment
    """
    return np.array([(y_coord + j, x_coord + k)
                     for j in range(-radius, radius + 1)
                      for k in range(-radius, radius + 1)]).transpose()


def calc_descriptor(im, pixel_environment_idx, k_dim):
    """
    This function is responsible for creating the final descriptor of a pixel
    :param im: the image that the pixel is in
    :param pixel_environment_idx: the environment of the pixel with some radius
    :return: the descriptor of the pixel
    """
    # map the coordinates of im in a new way according to the idx_env we get
    descriptor = (map_coordinates(im, pixel_environment_idx, order=1, prefilter=False)).reshape(k_dim, k_dim)
    # normalize the given descriptor with his mean
    descriptor = descriptor - np.mean(descriptor)
    if np.linalg.norm(descriptor) != 0:
        descriptor /= np.linalg.norm(descriptor)
    return descriptor


def sample_descriptor(im, pos, desc_rad):
    """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """
    # 1. iterate over each pixel and creates for him new environment around his
    # coordinates with the correspond radius
    k_dim = 1 + 2 * desc_rad
    n_dim = len(pos[:, 1])
    descriptors = np.empty((n_dim, k_dim, k_dim)) # the final to return
    for pixel in range(n_dim):
        pixel_environment_idx = calc_pixel_environment(pos[pixel][0], pos[pixel][1], desc_rad)
        # 2. calculate the final descriptor (map_coodrinates and normalize)
        descriptors[pixel, :, :] = calc_descriptor(im, pixel_environment_idx, k_dim)
    # 3. return the final result - an array of size (N, K, K)
    return descriptors


def find_features(pyr):
    """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
    # find the features points if the image
    feature_points = spread_out_corners(pyr[0], m=7, n=7, radius=7)
    descriptors = sample_descriptor(pyr[2], (2 ** (0 - 2)) * feature_points, RADIUS)
    return [feature_points, descriptors]


def calc_relations_matches(score, min_score):
    """
    This function checks the relations of the dot product to the other max values
    :param score: the dot product of two vectors (some descriptors)
    :param min_score: min value that we should pass
    :return: the final match points
    """
    match_points = score > min_score  # taking the points that larger then min
    if score.shape[1] > 1:  # cols
        second_max = np.partition(score, -2)[:, -2][:, np.newaxis]
        match_points = np.bitwise_and(match_points, score >= second_max)
    if score.shape[0] > 1:  # rows
        second_max = np.partition(score.transpose(), -2)[:, -2][:, np.newaxis]
        match_points = np.bitwise_and(match_points.transpose(), score.transpose() >= second_max).transpose()
    match_points = np.argwhere(match_points)
    return match_points


def match_features(desc1, desc2, min_score):
    """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """
    # check the input
    if desc1.size == 0 or desc2.size == 0:
        return [np.empty(0, dtype=np.int_), np.empty(0, dtype=np.int_)]
    # 1. calcs the dot product of the tow given descriptors arrays
    score = np.dot(desc1.reshape(desc1.shape[0], -1), desc2.reshape(desc2.shape[0], -1).transpose())
    # 2. comparing between the value Si,j = Di,j * Di+1,k to some max values
    #    and determine if there is a match between the 2 features
    match_points = calc_relations_matches(score, min_score)
    # 3. return list with [matching array for desc1, matching array for desc1]
    return [match_points[:, 0], match_points[:, 1]]


def apply_homography(pos1, H12):
    """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
    # moving to Homogeneous coordinates (add 1 to the original vec)
    homogen_vec = np.hstack([pos1, np.ones(pos1.shape[0])[:, np.newaxis]])
    # calc the dot product with H12
    dot_prod_vec = H12.dot(homogen_vec.transpose())
    # get z values for divide by them (zero division will be handle in ransac)
    result_homogen_vec = dot_prod_vec.transpose()
    z_coord_values = result_homogen_vec[:, -1][:, np.newaxis]
    final_vec = (result_homogen_vec / z_coord_values)
    # removing last column to get xy and return it
    return final_vec[:, :-1]


def calc_distance_from_matcher(points1_transform, points2, inlier_tol):
    """
    This function calc the distance between the new given points
    to the points2 array (squared euclidian distance), that way getting the
    inliers and the number of them
    :param points1_transform: the points who we apply the homography
    :param points2: the orig points in the other frame
    :param inlier_tol:  trashold
    :return: array of inliers points and the number of them
    """
    distance = (np.linalg.norm(points1_transform - points2, axis=1)) ** 2
    distance[distance < inlier_tol] = 0  # standing in the given trashold
    return distance, np.count_nonzero(distance == 0)


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
  Computes homography between two sets of points using RANSAC.
  :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """
    # check input
    if points1.size == 0 or (points1.shape[0] == 1 and not translation_only):
        return [np.eye(3, dtype=np.int_), np.array([])]
    max_inliers = 0
    pix1_max_inlier, pix2_max_inlier, inliers_indeces = None, None, None
    # iterate num_iter times
    for i in range(num_iter):
        # taking random 2 pixels in points1 and in points2
        random_indeces = np.random.randint(0, points1.shape[0], 2)
        pixel_pair1 = np.array([points1[random_indeces[0]], points1[random_indeces[1]]])
        pixel_pair2 = np.array([points2[random_indeces[0]], points2[random_indeces[1]]])
        cur_homography = estimate_rigid_transform(pixel_pair1, pixel_pair2, translation_only)
        points1_transform = apply_homography(points1, cur_homography)
        distance_from_points2, cur_inlier_num = calc_distance_from_matcher(points1_transform, points2, inlier_tol)
        if cur_inlier_num > max_inliers:   # compare the inlier points, larger num of inliers -> better result
            max_inliers = cur_inlier_num
            # calc all the indexes of the inliers (need to return indexes!)
            index_val = np.argwhere(distance_from_points2 == 0)
            inliers_indeces = index_val.reshape(index_val.shape[0], )
            pix1_max_inlier, pix2_max_inlier = pixel_pair1, pixel_pair2
    final_homography = estimate_rigid_transform(pix1_max_inlier, pix2_max_inlier, translation_only)
    # return a list with the final homography and inliers indexes
    return [final_homography, inliers_indeces]


def draw_lines(x_im1, x_im2, y_im1, y_im2, line_color):
    """
    This function draws lines between 4 groups of points, between 2 images. the
    first two groups of points is the groups of some type (inliers or outliers)
    in the first image and second image, respectively, in the x-axis. such that
    the other two images but in the y-axis.
    :param x_im1: x-axis points from some type from image1
    :param x_im2: x-axis points from some type from image2
    :param y_im1: y-axis points from some type from image1
    :param y_im2: y-axis points from some type from image2
    :param line_color: the color we want the line to be
    :return: None
    """
    for i in range(len(y_im1)):
        plt.plot([x_im1[i:i + 2], x_im2[i:i + 2]], [y_im1[i:i + 2],
        y_im2[i:i + 2]], mfc='r', c=line_color, lw=.3, ms=5, marker='.')


def display_matches(im1, im2, points1, points2, inliers):
    """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
    # concat the two given images and plot it
    concatenate_image = np.hstack([im1, im2])
    plt.imshow(concatenate_image, cmap='gray')
    # calc the indexes of inliers and outliers in each image
    outliers_indexes = np.setdiff1d(np.arange(0, points1.shape[0]), inliers)    # remember that inliers is a array of the inliers INDEXES
    im1_inliers, im1_outliers = points1[inliers], points1[outliers_indexes]
    im2_inliers, im2_outliers = points2[inliers], points2[outliers_indexes]
    # take the inliers of the x-axis and inliers of y-axis (same for outliers)
    x_inliers_im1, x_outliers_im1 = im1_inliers[:, 0], im1_outliers[:, 0]
    x_inliers_im2, x_outliers_im2 = im2_inliers[:, 0], im2_outliers[:, 0]
    y_inliers_im1, y_outliers_im1 = im1_inliers[:, 1], im1_outliers[:, 1]
    y_inliers_im2, y_outliers_im2 = im2_inliers[:, 1], im2_outliers[:, 1]
    # draw inliers lines
    draw_lines(x_inliers_im1, x_inliers_im2, y_inliers_im1, y_inliers_im2, 'y')
    # draw outliers lines
    draw_lines(x_outliers_im1, x_outliers_im2, y_outliers_im1, y_outliers_im2, 'b')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
  Convert a list of succesive homographies to a
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """
    h2m = [np.eye(3)]   # start from id matrix
    for i in range(m - 1, -1, -1):  # from m down
        cur_homography = h2m[0] @ H_succesive[i]
        cur_homography = cur_homography / cur_homography[2, 2]  # for H[2,2] = 1
        h2m.insert(0, cur_homography)
    for j in range(m, len(H_succesive)):    # from m up
        cur_homography = h2m[-1] @ np.linalg.inv(H_succesive[j])
        cur_homography = cur_homography / cur_homography[2, 2]
        h2m.append(cur_homography)
    return h2m


def compute_bounding_box(homography, w, h):
    """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
    top_left, top_right, bottom_left, bottom_right = [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]
    edge_points = np.array([top_left, top_right, bottom_left, bottom_right])
    edge_points_homography = apply_homography(edge_points, homography)
    edge_points_homography = np.round(edge_points_homography).astype(np.int_)
    edge_points_homography = np.sort(edge_points_homography, axis=0)
    return np.vstack([edge_points_homography[0], edge_points_homography[-1]])


def calc_pre_homography_matrix(bounding_box):
    top_left, bottom_right = bounding_box[0], bounding_box[1]
    x_matrix, y_matrix = np.meshgrid(np.arange(top_left[0], bottom_right[0] + 1), np.arange(top_left[1], bottom_right[1] + 1))
    return np.vstack([x_matrix.ravel(), y_matrix.ravel()]).transpose(), x_matrix.shape  # dim = shape for the future coordinates system of the warp image


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    # calc coordinates
    w, h = image.shape[1], image.shape[0]
    bounding_box = compute_bounding_box(homography, w, h)
    pre_homography_matrix, dim = calc_pre_homography_matrix(bounding_box)
    homography_grid = apply_homography(pre_homography_matrix, np.linalg.inv(homography))
    coordinates = [homography_grid[:, 1].reshape(dim), homography_grid[:, 0].reshape(dim)]
    # calc the warp image and return it
    back_warp_image = map_coordinates(image, coordinates, order=1, prefilter=False)
    return back_warp_image


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int_)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int_)
    ret = np.zeros_like(image, dtype=np.bool_)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int_)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int_)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int_)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret




def blend_low_high_res(low_res_image, low_res_feature_points, low_res_descriptors, high_res_image, high_res_feature_points, high_res_descriptors):
    # Match features between low-res and high-res images
    match_indices_low, match_indices_high = match_features(low_res_descriptors, high_res_descriptors, 0.7)

    # Filter out unmatched feature points
    matched_low_res_points = low_res_feature_points[match_indices_low]
    matched_high_res_points = high_res_feature_points[match_indices_high]

    # Estimate homography using RANSAC
    homography, inliers = ransac_homography(matched_low_res_points, matched_high_res_points, 1000, 10)

    # Warp low-res image using the estimated homography
    warped_low_res_image = warp_image(low_res_image, homography)

    # Blend warped low-res image with high-res image
    blended_image = blend_images(warped_low_res_image, high_res_image)

    return blended_image

def blend_images(image1, image2):
    # Blend images using simple averaging
    blended_image = (image1 + image2) / 2.0
    return blended_image

# Load low-resolution and high-resolution images and their corresponding parts
low_res_image = sol4_utils.read_image('low_res_image.jpg', 1)
high_res_image = sol4_utils.read_image('high_res_image.jpg', 2)
low_res_part = sol4_utils.read_image('low_res_part.jpg', 1)
high_res_part = sol4_utils.read_image('high_res_part.jpg', 2)

# Extract features from images
low_res_feature_points, low_res_descriptors = find_features([low_res_image, low_res_part])
high_res_feature_points, high_res_descriptors = find_features([high_res_image, high_res_part])

# Blend low-resolution and high-resolution parts
blended_image = blend_low_high_res(low_res_image, low_res_feature_points, low_res_descriptors, high_res_image, high_res_feature_points, high_res_descriptors)

# Display the blended image
plt.imshow(blended_image, cmap='gray')
plt.axis('off')
plt.show()


if __name__ == '__main__':
    print("Run - Go!")