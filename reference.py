import numpy as np


def orthographic_projection(geometry, color, precision, filtering):

    # geometry = np.asarray(cloud.points)
    # if np.any(cloud.get_min_bound() < 0):
    #     logging.info(f'Point Cloud {cloud} has negative points')
    #     geometry = geometry - cloud.get_min_bound()
    # if cloud.has_colors():
    #     color = np.asarray(cloud.colors)

    # color
    img_0 = np.ones(([2**precision, 2**precision, 3])) * 255
    img_1 = np.ones(([2**precision, 2**precision, 3])) * 255
    img_2 = np.ones(([2**precision, 2**precision, 3])) * 255
    img_3 = np.ones(([2**precision, 2**precision, 3])) * 255
    img_4 = np.ones(([2**precision, 2**precision, 3])) * 255
    img_5 = np.ones(([2**precision, 2**precision, 3])) * 255

    # occupancy map
    ocp_map_0 = np.zeros(([2**precision, 2**precision]))
    ocp_map_1 = np.zeros(([2**precision, 2**precision]))
    ocp_map_2 = np.zeros(([2**precision, 2**precision]))
    ocp_map_3 = np.zeros(([2**precision, 2**precision]))
    ocp_map_4 = np.zeros(([2**precision, 2**precision]))
    ocp_map_5 = np.zeros(([2**precision, 2**precision]))

    img = [img_0, img_1, img_2, img_3, img_4, img_5]
    ocp_map = [ocp_map_0, ocp_map_1, ocp_map_2, ocp_map_3, ocp_map_4, ocp_map_5]
    minDepth = np.zeros([3, 2**precision, 2**precision])
    maxDepth = np.ones([3, 2**precision, 2**precision]) * 2**precision

    plane = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

    for index in range(len(geometry)):
        if (geometry[index][0] >= 2**precision) or (geometry[index][1] >= 2**precision) or (geometry[index][2] >= 2**precision):
            continue
        else:
            for coord in range(0, 3):
                if geometry[index][coord] <= maxDepth[coord][geometry[index][plane[coord][0]].astype(np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)]:
                    img[2 * coord][geometry[index][plane[coord][0]].astype(
                        np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)][:] = color[index][:] * 255
                    ocp_map[2 * coord][geometry[index][plane[coord][0]
                                                       ].astype(np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)] = 1
                    maxDepth[coord][geometry[index][plane[coord][0]].astype(
                        np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)] = geometry[index][coord]

                if geometry[index][coord] >= minDepth[coord][geometry[index][plane[coord][0]].astype(np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)]:
                    img[2 * coord + 1][geometry[index][plane[coord][0]].astype(
                        np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)][:] = color[index][:] * 255
                    ocp_map[2 * coord + 1][geometry[index][plane[coord][0]
                                                           ].astype(np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)] = 1
                    minDepth[coord][geometry[index][plane[coord][0]].astype(
                        np.uint16)][geometry[index][plane[coord][1]].astype(np.uint16)] = geometry[index][coord]
    w = filtering
    c1 = c2 = c3 = c4 = c5 = c6 = 0
    if (w != 0):
        for i in range(w, 2**precision - w):
            for j in range(w, 2**precision - w):
                if (ocp_map[0][i, j] == 1) and (maxDepth[0][i, j] > sum(sum(maxDepth[0][i-w:i+w+1, j-w:j+w+1] * ocp_map[0][i-w:i+w+1, j-w:j+w+1]))/sum(sum(ocp_map[0][i-w:i+w+1, j-w:j+w+1])) + 20):
                    ocp_map[0][i, j] = 0
                    img[0][i, j, :] = 255
                    c1 += 1
                if (ocp_map[1][i, j] == 1) and (minDepth[0][i, j] < sum(sum(minDepth[0][i-w:i+w+1, j-w:j+w+1] * ocp_map[1][i-w:i+w+1, j-w:j+w+1]))/sum(sum(ocp_map[1][i-w:i+w+1, j-w:j+w+1])) - 20):
                    ocp_map[1][i, j] = 0
                    img[1][i, j, :] = 255
                    c2 += 1
                if (ocp_map[2][i, j] == 1) and (maxDepth[1][i, j] > sum(sum(maxDepth[1][i-w:i+w+1, j-w:j+w+1] * ocp_map[2][i-w:i+w+1, j-w:j+w+1]))/sum(sum(ocp_map[2][i-w:i+w+1, j-w:j+w+1])) + 20):
                    ocp_map[2][i, j] = 0
                    img[2][i, j, :] = 255
                    c3 += 1
                if (ocp_map[3][i, j] == 1) and (minDepth[1][i, j] < sum(sum(minDepth[1][i-w:i+w+1, j-w:j+w+1] * ocp_map[3][i-w:i+w+1, j-w:j+w+1]))/sum(sum(ocp_map[3][i-w:i+w+1, j-w:j+w+1])) - 20):
                    ocp_map[3][i, j] = 0
                    img[3][i, j, :] = 255
                    c4 += 1
                if (ocp_map[4][i, j] == 1) and (maxDepth[2][i, j] > sum(sum(maxDepth[2][i-w:i+w+1, j-w:j+w+1] * ocp_map[4][i-w:i+w+1, j-w:j+w+1]))/sum(sum(ocp_map[4][i-w:i+w+1, j-w:j+w+1])) + 20):
                    ocp_map[4][i, j] = 0
                    img[4][i, j, :] = 255
                    c5 += 1
                if (ocp_map[5][i, j] == 1) and (minDepth[2][i, j] < sum(sum(minDepth[2][i-w:i+w+1, j-w:j+w+1] * ocp_map[5][i-w:i+w+1, j-w:j+w+1]))/sum(sum(ocp_map[5][i-w:i+w+1, j-w:j+w+1])) - 20):
                    ocp_map[5][i, j] = 0
                    img[5][i, j, :] = 255
                    c6 += 1

    print("{t1} points removed from 1st view".format(t1=c1))
    print("{t1} points removed from 2nd view".format(t1=c2))
    print("{t1} points removed from 3rd view".format(t1=c3))
    print("{t1} points removed from 4th view".format(t1=c4))
    print("{t1} points removed from 5th view".format(t1=c5))
    print("{t1} points removed from 6th view".format(t1=c6))
    return (img, ocp_map)