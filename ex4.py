import cv2
import numpy as np
from skimage import io, color

def visualize_matches(image1, keypoints1, image2, keypoints2, matches):
    # Create a blank canvas to draw the merged image
    merged_image = np.zeros((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3), dtype=np.uint8)
    # Draw the images on the canvas
    merged_image[:image1.shape[0], :image1.shape[1]] = image1
    merged_image[:image2.shape[0], image1.shape[1]:] = cv2.cvtColor(image2, cv2.COLOR_BGRA2BGR)  # Convert to RGB
    # Draw keypoints on the merged image
    for match in matches:
        kp1 = keypoints1[match.queryIdx]
        kp2 = keypoints2[match.trainIdx]
        pt1 = (int(kp1.pt[0]), int(kp1.pt[1]))
        pt2 = (int(kp2.pt[0] + image1.shape[1]), int(kp2.pt[1]))
        cv2.circle(merged_image, pt1, 5, (0, 255, 0), -1)
        cv2.circle(merged_image, pt2, 5, (0, 255, 0), -1)
        cv2.line(merged_image, pt1, pt2, (0, 0, 255), 1)
    return merged_image


def visualize_keypoints_and_descriptors(image, keypoints, color_keypoints=(0, 255, 0), color_descriptors=(0, 0, 255)):
    # Draw circles at the keypoints' locations on the image
    vis_image = image.copy()
    for kp in keypoints:
        pt = (int(kp.pt[0]), int(kp.pt[1]))
        cv2.circle(vis_image, pt, 5, color_keypoints, -1)
        angle = kp.angle
        # Calculate the endpoint of the arrow indicating the orientation
        endpoint = (int(pt[0] + 30 * np.cos(angle * np.pi / 180)),
                    int(pt[1] + 30 * np.sin(angle * np.pi / 180)))
        # Draw the arrow
        cv2.arrowedLine(vis_image, pt, endpoint, color_descriptors, 2)
    return vis_image


def correct_perspective(image, reference_image, lake_flag=False):
    # Initialize SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=17)
    # Blur the image
    # blurred_image = cv2.GaussianBlur(image, (3,3),0)  # Kernel size (15x15) can be adjusted
    # cv2.imshow("blurred.jpg", blurred_image)
    # high_res_part_with_alpha = cv2.imread("lake_high_res.png", cv2.IMREAD_UNCHANGED)

    # if lake_flag:
    # image = cv2.GaussianBlur(gray_image, (3, 3), 0) # blurring the lake for better stiching!!
    # Find keypoints and descriptors in both images
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(reference_image, None)
    # # Visualize keypoints on the images
    # vis_low_res_img = visualize_keypoints_and_descriptors(reference_image, kp2)
    # vis_high_res_part_img = visualize_keypoints_and_descriptors(image, kp1)
    # cv2.imwrite("hr_vis.jpg", vis_high_res_part_img)
    # cv2.imwrite("lr_vis.png", vis_low_res_img)

    # # Display the images with keypoints
    # cv2.imshow('Low Resolution Image with Keypoints', vis_low_res_img)
    # cv2.imshow('High Resolution Part Image with Keypoints', vis_high_res_part_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Feature matching using FLANN (Fast Approximate Nearest Neighbor Search Library)
    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=20)    # works: checks=20, contrastThreshold=0.04, edgeThreshold=17
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # Ensure enough matches for RANSAC
    if len(good_matches) < 4:
        raise ValueError("Insufficient matches for RANSAC")
    # Find perspective transformation using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)

    # cv2.imshow('Blended Image', corrected_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Visualize keypoints and matches
    # merged_image = visualize_matches(low_res_img, kp1, high_res_part_img, kp2, good_matches)
    #
    # cv2.imwrite("merg_des_ratio.png", merged_image)

    # # Display the merged image
    # cv2.imshow('Keypoints and Matches', merged_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("warp_image.png", corrected_image)

    # Warp the image to correct perspective
    corrected_image = cv2.warpPerspective(image, M, (reference_image.shape[1], reference_image.shape[0]), flags=cv2.INTER_LINEAR)
    return corrected_image


def blend_images(low_res_img, high_res_part_img):
    # Correct perspective of high-res image
    high_res_corrected = correct_perspective(high_res_part_img, low_res_img)
    cv2.imwrite("output.png", high_res_corrected)
    # Read the high-resolution part image with alpha channel
    high_res_part_with_alpha = cv2.imread("output.png", cv2.IMREAD_UNCHANGED)
    # Threshold the alpha channel to create a binary mask
    _, binary_mask = cv2.threshold(high_res_part_with_alpha[:, :, 3], 128, 255, cv2.THRESH_BINARY)
    # Invert the binary mask
    inverted_mask = cv2.bitwise_not(binary_mask)
    # Resize the low-res image to match the high-res part image
    low_res_img_resized = cv2.resize(low_res_img, (high_res_part_with_alpha.shape[1], high_res_part_with_alpha.shape[0]))
    # Blend the images using the binary mask
    blended_img = cv2.bitwise_and(low_res_img_resized, low_res_img_resized, mask=inverted_mask)
    blended_img += cv2.bitwise_and(high_res_part_with_alpha[:, :, :3], high_res_part_with_alpha[:, :, :3], mask=binary_mask)
    return blended_img




if __name__ == "__main__":
    # Example 1
    # low_res_img = cv2.imread('desert_low_res.jpg')  # Read low-res image
    # high_res_part_img = cv2.imread('desert_high_res.png', cv2.IMREAD_UNCHANGED)  # Read high-res part image with alpha channel
    # # Perform alpha blending
    # result = blend_images(low_res_img, high_res_part_img)
    # # Display the blended image
    # cv2.imwrite("final_desert_result.png", result)
    # cv2.imshow('Blended Image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Example 2
    # low_res_img = cv2.imread('lake_low_res.jpg')  # Read low-res image
    # high_res_part_img = cv2.imread('lake_high_res.png', cv2.IMREAD_UNCHANGED)  # Read high-res part image with alpha channel
    # # Perform alpha blending
    # result = blend_images(low_res_img, high_res_part_img)
    # # Display the blended image
    # cv2.imwrite("final_lake_result.png", result)
    # cv2.imshow('Blended Image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pass
