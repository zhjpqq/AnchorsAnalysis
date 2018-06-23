bin_width = 1
bin_size = 10
bin_min = 0
bin_max = bin_min + bin_width * bin_size
point_arrs = np.zeros([bin_size, 2]).astype(np.float32)
point_arrs[:, 0] = np.arange(bin_min, bin_max, step=bin_width)


hotpoint:

hot_hitok_val = {1000:0.194, 3000:0.400, 6000: 0.598, 10000:0.765, 15000:0.801, 20000:0.867,
                 30000:0.946, 40000:0.983, 45000:, 50000:0.992, 80000:0.945, 100000:0.962, 200000: 999535}

hot_hitok_val_1W_3k_300 = {'C1': (0.663, 0.409, 0.103), 'C2': (0.688, 0.409,0.997), 'C3': (0.765,0.406, 0.087),
                           'C4': (, 0.288, 0.049), 'C5':( , , 0.051)}

uniform:

uniform_hitok_val = {1000:0.077, 3000:0.214, 6000: 0.391, 10000: 0.575, 15000: 0.750, 20000: 0.829,
                     30000:0.923, 40000: 0.968, 50000:0.983,  80000:0.986, 100000: 0.9998}

edges:

edges_hitok_val = {1000: 0.193, 2000: 0.314, 3000: 0.402, 4000: 0.472, 5000: 0.529, 6000: 0.569, 7000: 0.609,
                 8000: 0.631, 9000: 0.661, 10000: 0.680, 11000: 0.694, 12000: 0.712, 13000: 0.727, 14000: 0.0,
                 15000: 0.804, 16000: 0.758, 17000: 0.767, 18000: 0.775, 19000: 0.779, 20000: 0.783, 21000: 0.790,
                 22000: 0.792, 23000: 0., 25000:0.802, 27000: 0.806, 30000:0.877, 40000:0.895, 50000:0.933,
                 60000:0.938, 70000:0.952, 90000:0.956, 110000:0.956, 160000:0.956, 200000:0.957, 300000:0.957,
                 500000: 0.995}

edges_hitok_val = {1000:0.193, 3000:0.402, 6000: 0.569, 10000: 0.680, 15000: 0.804, 20000: 0.783,
                   30000:0.877, 40000: 0.895, 50000:0.933,  80000:0.954, 100000: 0.956}

edges=[10, 20]

nums, rate = (1W, 0.688) (2W, 0.889) (3W, 0.950) (5W, 0.983) (8W, 0.989) (10W, 0.9897)

edges = [10, 200]

nums, rate = (1W, ) (2W, 0.809) (3W, 0.847)

edges = [10, 280]
nums, rate = (1W, ) (2W, ) (3W, )

edges = [30, 100]
nums, rate = (1W, ) (2W, 0.840) (3W, )

edges = [60, 200]
nums, rate = (1W, ) (2W, 0.784) (3W, )




fig1 = plt.figure()
ax11 = fig1.add_subplot(2, 1, 1)
ax12 = fig1.add_subplot(2, 1, 2)
ax11.hist(best_match_dist, bins=200, range=(0, 60), normed=False)
ax12.hist(best_match_dist, bins=200, range=(0, 60), normed=True)
ax11.set_title('data: %s, method: %s, nums: %s, dist-mean: %s' % (
    'val', config.ANCHOR_METHOD, config.ANCHORS_PER_IMAGE, best_match_mean))
# fig1.show()

fig2 = plt.figure()
ax21 = fig2.add_subplot(2, 1, 1)
ax22 = fig2.add_subplot(2, 1, 2)
ax21.hist(hitok_rate, bins=20, normed=False)  # (1-0)/20 = 0.05
ax22.hist(gtbox_nums, bins=20, normed=True)
ax21.set_title('hitok rate mean : %s' % hitok_mean)

plt.show()
