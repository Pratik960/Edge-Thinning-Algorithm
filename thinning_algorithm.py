import cv2
import numpy as np
from scipy.ndimage import sobel
import matplotlib.pyplot as plt


input_image = cv2.imread("edgest.bmp", cv2.IMREAD_GRAYSCALE)
# cv2.imwrite("gray.bmp", input_image)
height, width = input_image.shape
iteration = 0
edge_points = set()
output_image = np.zeros((height, width), dtype=np.uint8)
first_edge_point = None
patterns = {
   'horizontal_a': np.array([[0, 0, 0],
                             [1, 2, 3],
                             [0, 0, 0]]),  # Pattern 1a
    'horizontal_b': np.array([[0, 0, 0],
                              [3, 2, 1],
                              [0, 0, 0]]),  # Pattern 1b
    'vertical_a' : np.array([[0, 1, 0],
                             [0, 2, 0],
                             [0, 3, 0]]),  # Pattern 2a
    'vertical_b' : np.array([[0, 3, 0],
                             [0, 2, 0],
                             [0, 1, 0]]),  # Pattern 2b
    'gradient_45_a' : np.array([[0, 0, 1],
                                [0, 2, 0],
                                [3, 0, 0]]),  # Pattern 3a
    'gradient_45_b' : np.array([[0, 0, 1],
                                [0, 2, 0],
                                [0, 3, 0]]),  # Pattern 3b
    'gradient_45_c' : np.array([[0, 0, 1],
                                [3, 2, 0],
                                [0, 0, 0]]),  # Pattern 3c
    'gradient_45_d' : np.array([[0, 1, 0],
                                [0, 2, 0],
                                [3, 0, 0]]),  # Pattern 3d
    'gradient_45_e' : np.array([[0, 0, 0],
                                [0, 2, 1],
                                [3, 0, 0]]),  # Pattern 3e
    'gradient_45_f' : np.array([[0, 0, 3],
                                [0, 2, 0],
                                [1, 0, 0]]),  # Pattern 3f
    'gradient_45_g' : np.array([[0, 0, 3],
                                [0, 2, 0],
                                [0, 1, 0]]),  # Pattern 3g
    'gradient_45_h' : np.array([[0, 0, 3],
                                [1, 2, 0],
                                [0, 0, 0]]),  # Pattern 3h
    'gradient_45_i' : np.array([[0, 3, 0],
                                [0, 2, 0],
                                [1, 0, 0]]),  # Pattern 3i
    'gradient_45_j' : np.array([[0, 0, 0],
                                [0, 2, 3],
                                [1, 0, 0]]),  # Pattern 3j
    'gradient_135_a' : np.array([[1, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 3]]),  # Pattern 4a
    'gradient_135_b' : np.array([[1, 0, 0],
                                 [0, 2, 0],
                                 [0, 3, 0]]),  # Pattern 4b
    'gradient_135_c' : np.array([[1, 0, 0],
                                 [0, 2, 3],
                                 [0, 0, 0]]),  # Pattern 4c
    'gradient_135_d' : np.array([[0, 1, 0],
                                 [0, 2, 0],
                                 [0, 0, 3]]),  # Pattern 4d
    'gradient_135_e' : np.array([[0, 0, 0],
                                 [1, 2, 0],
                                 [0, 0, 3]]),  # Pattern 4e
    'gradient_135_f' : np.array([[3, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 1]]),  # Pattern 4f
    'gradient_135_g' : np.array([[3, 0, 0],
                                 [0, 2, 0],
                                 [0, 1, 0]]),  # Pattern 4g
    'gradient_135_h' : np.array([[3, 0, 0],
                                 [0, 2, 1],
                                 [0, 0, 0]]),  # Pattern 4h
    'gradient_135_i' : np.array([[0, 3, 0],
                                 [0, 2, 0],
                                 [0, 0, 1]]),  # Pattern 4i
    'gradient_135_j' : np.array([[0, 0, 0],
                                 [3, 2, 0],
                                 [0, 0, 1]]),  # Pattern 4j
}



















def find_matching_matrix(detected_pattern, patterns):
    for pattern_name, pattern_matrix in patterns.items():
        if np.array_equal(detected_pattern, pattern_matrix):
            return pattern_name
    return None

def get_neighborhood_max_position(image, i, j):
    rows, cols = image.shape

    max_value = -np.inf
    max_position = None

    for x in range(max(0, i-1), min(i+2, rows)):
        for y in range(max(0, j-1), min(j+2, cols)):
            if (x, y) != (i, j) and image[x, y] > max_value:
              max_value = image[x, y]
              max_position = (x,y)

    return max_position

def resolve_with_gradient(input_image, candidate_points):
    """
    Use Sobel operator to resolve P3 among candidates.

    Args:
    - image: The original image matrix.
    - candidates: List of candidate positions.

    Returns:
    - The selected position of P3.
    """
    # Compute gradients using Sobel operator
    sobel_x = sobel(input_image, axis=1)
    sobel_y = sobel(input_image, axis=0)

    max_magnitude = -1
    selected_p3 = None

    for row, col in candidate_points:

        # Calculate gradient magnitude
        gx = sobel_x[row, col]
        gy = sobel_y[row, col]
        magnitude = np.sqrt(gx**2 + gy**2)

        # Update the point with the maximum gradient
        if magnitude > max_magnitude:
            max_magnitude = magnitude
            selected_p3 = (row, col)

    return selected_p3



def resolve_with_gradient_for_Ni(input_image, candidate_points, P3):
    """
    Use Sobel operator to resolve next edge point among candidates.

    Args:
    - image: The original image matrix.
    - candidates: List of candidate positions.

    Returns:
    - The selected position of Next edge point.
    """
    # Compute gradients using Sobel operator
    sobel_x = sobel(input_image, axis=1)
    sobel_y = sobel(input_image, axis=0)

    i, j = P3
    rows, cols = len(input_image), len(input_image[0])

    # total_direction = 0
    # count = 0

    # # Iterate through the neighbors, excluding the center (P3 itself)
    # for ni in range(i-1, i+2):
    #     for nj in range(j-1, j+2):
    #         if (ni, nj) != (i, j):  # Skip P3 (center)
    #             if 0 <= ni < rows and 0 <= nj < cols:
    #                 # Compute gradient for each neighbor
    #                 gx = sobel_x[ni, nj]
    #                 gy = sobel_y[ni, nj]
    #                 # magnitude = np.sqrt(gx**2 + gy**2)
    #                 direction = np.arctan2(gy, gx)

    #                 total_direction += direction
    #                 count += 1

    # mean_direction = total_direction / count if count > 0 else 0


    total_direction = 0
    total_weight = 0  # For weighted averaging

    # Iterate through the neighbors, excluding the center (P3 itself)
    for ni in range(i - 1, i + 2):
        for nj in range(j - 1, j + 2):
            if (ni, nj) != (i, j) and 0 <= ni < rows and 0 <= nj < cols:
                # Compute gradient for each neighbor
                gx = sobel_x[ni, nj]
                gy = sobel_y[ni, nj]
                magnitude = np.sqrt(gx**2 + gy**2)  # Gradient magnitude
                if magnitude > 0:  # Ignore zero gradients to avoid noise
                    direction = np.arctan2(gy, gx)
                    total_direction += direction * magnitude
                    total_weight += magnitude

    # Calculate weighted mean direction
    mean_direction = total_direction / total_weight if total_weight > 0 else 0
    print("Mean direction for next edge point is ", mean_direction)

    T = np.tan(mean_direction)
    res_direction = None

    if abs(T) > 2.4142: # Horizontal
        res_direction = 'horizontal'
    elif -0.4142 < T < 0.4142:  # Vertical
        res_direction = 'vertical'
    elif 0.4142 < T < 2.4142:  # 45° diagonal
        res_direction = '45 degrees'
    elif -2.4142 < T < -0.4142:  # 135° diagonal
        res_direction = '135 degrees'

    print("Result direction is : ", res_direction)
    print("Candidate points for Ni is :" , candidate_points)
    next_edge_point = None
    p3_row, p3_col = P3

    if res_direction == 'horizontal':
        #from point p3 get its right side pixel and left side pixel position and check if it exists in n1_n2_n3_positions
        left_pixel = (p3_row, p3_col - 1)
        right_pixel = (p3_row, p3_col + 1)

        # Check if these positions exist in candidate points
        if left_pixel in candidate_points:
            next_edge_point = left_pixel
        elif right_pixel in candidate_points:
            next_edge_point = right_pixel


    elif res_direction == 'vertical':
        #from point p3 get its above pixel and below pixel position and check if it exists in n1_n2_n3_positions
        # Get above and below pixel positions
        above_pixel = (p3_row - 1, p3_col)
        below_pixel = (p3_row + 1, p3_col)

        # Check if these positions exist in candidate_points
        if above_pixel in candidate_points:
            next_edge_point = above_pixel
        elif below_pixel in candidate_points:
            next_edge_point = below_pixel


    elif res_direction == '45 degrees' or  res_direction == '135 degrees':
        #from point p3 get row ,col offset (-1,1) and row, col offset (1,-1) and check if it exists in n1_n2_n3_positions
        diagonal_up_right = (p3_row - 1, p3_col + 1)
        diagonal_down_left = (p3_row + 1, p3_col - 1)

        #from point p3 get row ,col offset (-1,-1) and row, col offset (1,1) and check if it exists in n1_n2_n3_positions
        diagonal_up_left = (p3_row - 1, p3_col - 1)  # Diagonal position (up-left)
        diagonal_down_right = (p3_row + 1, p3_col + 1)  # Diagonal position (down-right)

        # Check if these positions exist in candidate_points
        if diagonal_up_right in candidate_points:
            next_edge_point = diagonal_up_right
        elif diagonal_down_left in candidate_points:
            next_edge_point = diagonal_down_left
        elif diagonal_up_left in candidate_points:
            next_edge_point = diagonal_up_left
        elif diagonal_down_right in candidate_points:
            next_edge_point = diagonal_down_right


    return next_edge_point







def find_point_p3(input_image, P1, P2):
    center_row, center_col = P2
    p1_row, p1_col = P1
    candidate_points = []

    # Create a new 3x3 matrix initialized with zeros
    neighborhood = np.zeros((3, 3), dtype=int)

    neighborhood[1, 1] = 2

    # Calculate the relative position of P1 with respect to P2
    relative_row_p1 = p1_row - center_row + 1
    relative_col_p1 = p1_col - center_col + 1

    if 0 <= relative_row_p1 < 3 and 0 <= relative_col_p1 < 3:
        neighborhood[relative_row_p1, relative_col_p1] = 1


    # Put P3 in row at different column to find candidate points for P3
    for i in range(3):
        for j in range(3):
            # print("Value of I and J are : ", i, j)
            if neighborhood[i, j] != 0:
                continue
            neighborhood[i, j] = 3

            matching_pattern_name = find_matching_matrix(neighborhood, patterns)
            # Check if the modified matrix matches any pattern
            if matching_pattern_name is not None:
                print("Matching pattern is : ", matching_pattern_name)
                original_row = P2[0] - 1 + i
                original_col = P2[1] - 1 + j
                if 0 <= original_row < input_image.shape[0] and 0 <= original_col < input_image.shape[1]:
                    # print("Added candidate points : ", i, j)
                    candidate_points.append((original_row, original_col))  # Store the position

            neighborhood[i, j] = 0

    if not candidate_points:
        return None
    print("Candidate poits for Point P3 are : " , candidate_points)

    max_value = float('-inf')
    max_positions = []
    for point in candidate_points:
        value = input_image[point[0], point[1]]
        if value > max_value:
            max_value = value
            max_positions = [(point[0], point[1])]
        elif value == max_value:
            max_positions.append((point[0], point[1]))

    if len(max_positions) > 1:
        print("Resolving with gradient for Point P3")
        return resolve_with_gradient(input_image, candidate_points)
    else:
        return max_positions[0] if max_positions else None


def find_next_edge_point(input_image,P2,P3):
    center_row, center_col = P3
    p2_row, p2_col = P2
    candidate_points = []

    # Create a new 3x3 matrix initialized with zeros
    neighborhood = np.zeros((3, 3), dtype=int)

    neighborhood[1, 1] = 2

    # Calculate the relative position of P2 with respect to P3
    relative_row_p2 = p2_row - center_row + 1
    relative_col_p2 = p2_col - center_col + 1

    if 0 <= relative_row_p2 < 3 and 0 <= relative_col_p2 < 3:
        neighborhood[relative_row_p2, relative_col_p2] = 1

    print("Neighborhood when deciding next edge point : " ,neighborhood)

    for i in range(3):
        for j in range(3):
            if neighborhood[i, j] == 0:  # Check for zeros
                neighborhood[i , j] = 3

                # Check if the modified matrix matches any pattern
                if find_matching_matrix(neighborhood, patterns) is not None:
                    original_row = P3[0] - 1 + i
                    original_col = P3[1] - 1 + j
                    if 0 <= original_row < input_image.shape[0] and 0 <= original_col < input_image.shape[1]:
                        candidate_points.append((original_row, original_col))  # Store the position

                neighborhood[i ,j] = 0

    if not candidate_points:
        return None, []
    print("Candidate points for deciding next edge point : ", candidate_points)

    candidate_points.sort(key=lambda pt: input_image[pt[0], pt[1]], reverse=True)
    if len(candidate_points) > 1 and input_image[candidate_points[0][0], candidate_points[0][1]] == input_image[candidate_points[1][0], candidate_points[1][1]]:
        print("Resolving with gradient for Point Ni")
        return resolve_with_gradient_for_Ni(input_image, candidate_points, P3), candidate_points
    else:
        return candidate_points[0] if candidate_points else None, candidate_points





def get_max_edge_width(image):
    # Ensure image is binary
    _, image = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels_im = cv2.connectedComponents(image)

    # Use distance transform on the entire binary image once
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 0)

    max_width = 0
    # Iterate over each label (excluding the background)
    for label in range(1, num_labels):
        # Create a mask for the current component
        component_mask = (labels_im == label)

        # Get the maximum value of the distance transform for this component
        width = np.max(dist_transform[component_mask]) * 2  # Full width is twice the distance
        max_width = max(max_width, width)

    return max_width




def check_boundary(image,d,first_edge_point, p1, p2, p3, n1_n2_n3_positions):

    flag_E = False
    flag_B = False
    flag_C = False

    #check if end point
    if all(
        (0 > n[0] or n[0] >= image.shape[0] or
         0 > n[1] or n[1] >= image.shape[1] or
         image[n[0], n[1]] == 0)
        for n in n1_n2_n3_positions
    ):
        flag_E = True


    # Check if boundary
    d = d / 2
    print("P3 point in boundary cheking is ", p3)
    Nr, Nc = p3
    M, N = image.shape
    if (Nr<=d and Nc <= d) or ((Nr>=M-d) or (Nc >= N-d)):
      flag_B = True


    #Check closed edge
    # D = abs(x2 - x1) + abs(y2 - y1)
    S = first_edge_point
    N = p3
    Dns =  abs(N[0] - S[0]) + abs(N[1] - S[1])
    Dp1s = abs(p1[0] - S[0]) + abs(p1[1] - S[1])
    Dp2s = abs(p2[0] - S[0]) + abs(p2[1] - S[1])
    Dp3s = abs(p3[0] - S[0]) + abs(p3[1] - S[1])
   
    if ( (Dp3s <= Dp2s) and (Dp3s <= Dp1s) and (Dp3s <= Dns)):
      flag_C = True

    return (flag_E or flag_B or flag_C)



def mask_operation(input_image ,d,edge_points):
    print("Masking operation started...........")
    rows, cols = len(input_image), len(input_image[0])
    # d = int(get_max_edge_width(input_image))

    print("Max edge width is : ", d)

    if d % 2 == 0:
        d = d + 1

    for i, j in edge_points:
        # Precompute the bounds to avoid boundary checks inside the loops
        row_min = max(i - d, 0)
        row_max = min(i , rows - 1)
        col_min = max(j - d, 0)
        col_max = min(j , cols - 1)
        # print("Min and Max Row : ", row_min, row_max)
        # print("Min and Max Col : ", col_min, col_max)

        # Traverse the range of maximum width d around the edge point
        for ni in range(row_min, row_max + 1):
            for nj in range(col_min, col_max + 1):
                # if 0 <= ni < rows and 0 <= nj < cols:
                  input_image[ni, nj] = 0

    print("Masking operation completed...........")
    return input_image


def check_if_match_pattern(P1, P2, P3):
    # Initialize the 3x3 matrix with zeros
    neighborhood = np.zeros((3, 3), dtype=int)

    neighborhood[1, 1] = 2

    # Get relative positions of P1 and P3 based on P2
    p1_row, p1_col = P1
    p2_row, p2_col = P2
    p3_row, p3_col = P3

    # Relative position of P1 in the 3x3 matrix
    relative_p1_row = p1_row - p2_row + 1
    relative_p1_col = p1_col - p2_col + 1

    # Relative position of P3 in the 3x3 matrix
    relative_p3_row = p3_row - p2_row + 1
    relative_p3_col = p3_col - p2_col + 1

    if 0 <= relative_p1_row < 3 and 0 <= relative_p1_col < 3:
        neighborhood[relative_p1_row, relative_p1_col] = 1

    if 0 <= relative_p3_row < 3 and 0 <= relative_p3_col < 3:
        neighborhood[relative_p3_row, relative_p3_col] = 3

    print("Detected Pattern when matching pattern : " , neighborhood)
    return find_matching_matrix(neighborhood, patterns) is not None


def fill_black_pixels_and_contours(image):
    # Step 1: Fill small black regions using closing
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return closing







Num_M = 0
# input_image = fill_black_pixels_and_contours(input_image)
# cv2.imwrite("Filled.bmp", input_image)
max_edge_width = int(get_max_edge_width(input_image))
print("Image shape is : ", input_image.shape)
for i in range(height):
  for j in range(width):
    P1 = None
    P2 = None
    P3 = None
    iteration = iteration + 1

    if input_image[i,j] != 0:
            # Noise point checking
            if input_image[i,j] < 30:
              input_image[i,j] = 0
              continue

            print("Iteration : " , iteration)

            P1 = (i,j)
            if first_edge_point is None:
                first_edge_point = P1

            # Finding P2
            P2 = get_neighborhood_max_position(input_image, P1[0], P1[1])
            print("P1 and P2 are : ", P1, P2)
            if P2 is not None:
                # Finding P3
                P3 = find_point_p3(input_image, P1, P2)

                if P3 is None:
                  print("No valid P3 found, stopping iteration.")
                  continue

                if not check_if_match_pattern(P1, P2, P3):
                  continue

                print("Point P3 is : " , P3)
                output_image[P1[0], P1[1]] = 255
                output_image[P2[0], P2[1]] = 255
                output_image[P3[0], P3[1]] = 255

                edge_points.update([P1, P2, P3])

                # Find next edge point
                next_edge_point, n1_n2_n3_positions = find_next_edge_point(input_image,P2,P3)

                print("Next edge point is : ", next_edge_point)

                if next_edge_point is None:
                  continue

                # Location exchange
                P1 = P2
                P2 = P3
                P3 = next_edge_point

                # Record these location Pending task
                output_image[P1[0], P1[1]] = 255
                output_image[P2[0], P2[1]] = 255
                output_image[P3[0], P3[1]] = 255

                edge_points.update([P1, P2, P3])

                while(not check_boundary(input_image, max_edge_width, first_edge_point, P1, P2, P3, n1_n2_n3_positions)):
                     next_edge_point, n1_n2_n3_positions = find_next_edge_point(input_image,P2,P3)
                     print("Next edge point is : ", next_edge_point)

                     if next_edge_point is None:
                        break

                     # Location exchange
                     P1 = P2
                     P2 = P3
                     P3 = next_edge_point

                     # Record these location Pending task
                     output_image[P1[0], P1[1]] = 255
                     output_image[P2[0], P2[1]] = 255
                     output_image[P3[0], P3[1]] = 255

                     edge_points.update([P1, P2, P3])


                # Check if P3 is a boundary or end point
                # if check_boundary(input_image, max_edge_width, first_edge_point, P1, P2, P3, n1_n2_n3_positions):
                # Perform mask operation
                mask_operation(input_image, max_edge_width, edge_points)
                Num_M = Num_M + 1
                print("Mask Operation Number : ", Num_M)
                first_edge_point = None


cv2.imwrite("Thinned_image.bmp", output_image)

