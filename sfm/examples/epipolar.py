#!/usr/bin/env python

'''
Feature-based image matching sample.

Note, that you will need the https://github.com/opencv/opencv_contrib repo for SIFT and SURF

USAGE
  find_obj.py [--feature=<sift|surf|orb|akaze|brisk>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf, orb or brisk. Append '-flann'
               to feature name to use Flann-based matcher instead bruteforce.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
from common import anorm, getsize

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6


def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        raise ValueError
        detector = cv2.xfeatures2d.SIFT_create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        raise ValueError
        detector = cv2.xfeatures2d.SURF_create(800)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(400)
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)


def explore_match(img1, img2, kp_pairs, status=None, H=None):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imwrite('./matching.png', vis)
    return


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


if __name__ == '__main__':
    print(__doc__)

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'brisk')
    try:
        fn1, fn2 = args
    except Exception as e:
        print(e)
        raise ValueError

    img1 = cv2.imread(cv2.samples.findFile(fn1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(cv2.samples.findFile(fn2), cv2.IMREAD_GRAYSCALE)
    detector, matcher = init_feature(feature_name)

    if img1 is None:
        print('Failed to load fn1:', fn1)
        sys.exit(1)

    if img2 is None:
        print('Failed to load fn2:', fn2)
        sys.exit(1)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)

    print('using', feature_name)

    # 特徴点の抽出
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

    print('matching...')
    # 特徴点のマッチング
    # 各特徴点に対する上位2つの点を返す
    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    if len(p1) >= 4:
        # ホモグラフィ行列とマスク
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
    else:
        H, status = None, None
        print('%d matches found, not enough for homography estimation' % len(p1))

    # 特徴点のマッチング
    explore_match(img1, img2, kp_pairs, status, H)

    # Fundamental Matrixを計算
    p1 = np.int32(p1)
    p2 = np.int32(p2)
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_LMEDS)

    # We select only inlier points
    # インライアの特徴点を抽出
    p1 = p1[mask.ravel() == 1]
    p2 = p2[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    # 画像2上の特徴点に対応するエピポーラを画像1上に描画する
    l1 = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2), 2, F)
    l1 = l1.reshape(-1, 3)
    img1_line, _ = drawlines(img1, img2, l1, p1, p2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    # 画像2上の特徴点に対応するエピポーラを画像1上に描画する
    l2 = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, F)
    l2 = l2.reshape(-1, 3)
    img2_line, _ = drawlines(img2, img1, l2, p2, p1)

    # エピポーラも合わせた画像の表示
    frame = cv2.hconcat([img1_line, img2_line])
    cv2.imwrite('epipole.png', frame)

    cv2.waitKey()
    cv2.destroyAllWindows()
