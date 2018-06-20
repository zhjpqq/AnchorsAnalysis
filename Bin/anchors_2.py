bin_width = 1
bin_size = 10
bin_min = 0
bin_max = bin_min + bin_width * bin_size
point_arrs = np.zeros([bin_size, 2]).astype(np.float32)
point_arrs[:, 0] = np.arange(bin_min, bin_max, step=bin_width)

edges:


hitok = {1000: 0.193, 2000: 0.314, 3000: 0.402, 4000: 0.472, 5000: 0.529, 6000: 0.569, 7000: 0.609,
         8000: 0.631, 9000: 0.661, 10000: 0.680, 11000: 0.694, 12000: 0.712, 13000: 0.8, 14000: 0.8, 15000: 0.8, 16000: 0.8}
