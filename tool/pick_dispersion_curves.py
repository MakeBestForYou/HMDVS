#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import argparse
from collections import defaultdict
import sys
from scipy.ndimage.filters import gaussian_filter


class Pick:
    def __init__(self, fig, ax, f, c, spectrum, width, nmax, result):
        self.fig = fig
        self.ax = ax
        self.spectrum = spectrum
        self.f = f
        self.c = c
        self.df = f[1] - f[0]
        self.dc = c[1] - c[0]
        self.half_width = width // 2
        self.nmax = nmax

        self.store = result
        self.buf = []
        self.current_mode = -1
        self.prev_mode = -2
        self.state = False  # True when starting to pick and versa.

    def connect(self):
        self.cidpress = (self.fig.canvas.mpl_connect
                         ("key_press_event", self.on_press))
        self.cidclick = (self.fig.canvas.mpl_connect
                         ("button_press_event", self.on_click))


    def find_idx_max(self, x, y):
        idx_f0 = int(np.round((x - self.f[0]) / self.df))
        idx_c0 = int(np.round((y - self.c[0]) / self.dc))
        value_range = self.spectrum[idx_c0 - self.half_width:
                                    idx_c0 + self.half_width,
                                    idx_f0]
        idx_max_range = np.argmax(value_range)
        idx_c_pick = idx_c0 - self.half_width + idx_max_range
        return (idx_f0, idx_c_pick)


    def find_cmax(self, x, y):
        idx_f0 = int(np.round((x - self.f[0]) / self.df))
        idx_c0 = int(np.round((y - self.c[0]) / self.dc))
        value_range = self.spectrum[idx_c0 - self.half_width:
                                    idx_c0 + self.half_width,
                                    idx_f0]
        idx_nmax = np.argsort(value_range)[::-1][:self.nmax]
        idx_c_nmax = idx_c0 - self.half_width + idx_nmax
        v_nmax = value_range[idx_nmax]
        vmax_nmax = np.sum(v_nmax)
        c_nmax = self.c[idx_c_nmax]
        c_avg = np.sum(c_nmax * v_nmax / vmax_nmax)
        return (idx_f0, c_avg)



    def on_click(self, event):
        if self.current_mode == -1:
            info = "Warning: give the current mode before the first pick."
            self.ax.set_title(info)
            self.fig.canvas.draw()
            return
        x = event.xdata
        y = event.ydata
        # idx_max = self.find_idx_max(x, y)
        # self.ax.plot(self.f[idx_max[0]], self.c[idx_max[1]], 'k.')
        # self.buf.append(idx_max)
        idx_f, c = self.find_cmax(x, y)
        self.ax.plot(self.f[idx_f], c, 'k.')
        self.buf.append((idx_f, c))

    def on_press(self, event):
        valid_range = ["{:d}".format(i) for i in range(10)]
        if event.key in valid_range:
            self.current_mode = int(event.key)
            if self.prev_mode < 0:
                self.prev_mode = self.current_mode
            self.ax.set_title("current mode: {:d}".format(self.current_mode))
            self.fig.canvas.draw()
        else:
            info = "Warning: the key pressed is invalid"
            self.ax.set_title(info)
            self.fig.canvas.draw()
            return
        if self.prev_mode != self.current_mode:
            self.prev_mode = self.current_mode
            self.state = True
            self.buf = []
            return

        # for the case when pick another region of the same mode
        if not self.state:
            self.state = True
            self.buf = []

        buf = sorted(set(self.buf))
        npick = len(buf)
        if npick <= 1:
            return

        f_plot = []
        c_plot = []
        for i in range(npick-1):
            # idx_f1, idx_c1 = buf[i][0], buf[i][1]
            # f1, c1 = self.f[idx_f1], self.c[idx_c1]
            # idx_f2, idx_c2 = buf[i+1][0], buf[i+1][1]
            # f2, c2 = self.f[idx_f2], self.c[idx_c2]
            # func_line = lambda x: (c2 - c1) / (f2 - f1) * (x - f1) + c1

            # j = idx_f1
            # self.store[self.prev_mode].append(self.buf[i])
            # while j < idx_f2:
            #     j += 1
            #     x = f[j]
            #     y = func_line(x)
            #     idx_max = self.find_idx_max(x, y)
            #     f_plot.append(self.f[idx_max[0]])
            #     c_plot.append(self.c[idx_max[1]])
            #     self.store[self.prev_mode].append(idx_max)
            idx_f1, c1 = buf[i][0], buf[i][1]
            f1 = self.f[idx_f1]
            idx_f2, c2 = buf[i+1][0], buf[i+1][1]
            f2 = self.f[idx_f2]
            func_line = lambda x: (c2 - c1) / (f2 - f1) * (x - f1) + c1

            j = idx_f1
            self.store[self.prev_mode].append(self.buf[i])
            while j < idx_f2:
                j += 1
                x = f[j]
                y = func_line(x)
                idx_f, c = self.find_cmax(x, y)
                f_plot.append(self.f[idx_f])
                c_plot.append(c)
                self.store[self.prev_mode].append((idx_f, c))
        self.ax.plot(f_plot, c_plot, 'k.')
        self.fig.canvas.draw()
        print("mode {:d}".format(self.prev_mode))
        for x in self.store[self.prev_mode]:
            print("{:9.3f}{:9.3f}".format(self.f[x[0]], x[1]))
        print("-"*50)
        print()
        self.state = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pick the dispersion"+
                                     " curves from data")
    parser.add_argument("--file_f",
                        default="f.txt",
                        help="input file for frequency[f.txt]")
    parser.add_argument("--file_c",
                        default="c.txt",
                        help="input file for phase velocity[c.txt]")
    parser.add_argument("--file_fc",
                        default="fc.txt",
                        help="input file for dispersion spectrum[fc.txt]")
    parser.add_argument("--width",
                        default=20,
                        type=int,
                        help="the width(number of points)"+
                        " of window to pick the maximum[20]")
    parser.add_argument("--plot",
                        default=None,
                        help="data file to load for plotting")
    parser.add_argument("--plot_data",
                        action="store_true",
                        help="only plotting dispersion spectrum")
    parser.add_argument("--unit_km",
                        action="store_true",
                        help="unit is kilometer for phase velocity")
    parser.add_argument("--nmax",
                        type=int,
                        default=3,
                        help="number of maximum to average")
    parser.add_argument("--cmax",
                        type=float,
                        default=None,
                        help="cmax for plotting")
    parser.add_argument("--cmin",
                        type=float,
                        default=None,
                        help="cmin for plotting")
    parser.add_argument("--fmax",
                        type=float,
                        default=None,
                        help="fmax for plotting")
    parser.add_argument('--npy',
                        action='store_true',
                        help='data in npy format')
    args = parser.parse_args()
    file_f = args.file_f
    file_c = args.file_c
    file_fc = args.file_fc
    width = args.width
    file_plot = args.plot
    plot_data = args.plot_data
    unit_km = args.unit_km
    nmax = args.nmax
    cmin = args.cmin
    cmax = args.cmax
    fmax = args.fmax
    use_npy = args.npy

    # load data
    if use_npy:
        f = np.load(file_f).reshape(-1,)
        c = np.load(file_c).reshape(-1,)
        spectrum = np.load(file_fc)
    else:
        f = np.loadtxt(file_f)
        c = np.loadtxt(file_c)
        spectrum = np.loadtxt(file_fc)

    if not unit_km:
        c = c / 1000.

    nf = len(f)
    nc = len(c)
    if spectrum.shape == (nc, nf):
        spectrum = spectrum.T

    spectrum = gaussian_filter(spectrum, 1)

    if cmax is None:
        cmax = np.max(c)
    if cmin is None:
        cmin = np.min(c)
    if fmax is None:
        fmax = np.max(f)

    # regularization
    for x in spectrum:
        x /= np.amax(np.abs(x))
    spectrum = spectrum.T

    if file_plot:
        data = np.loadtxt(file_plot)
        plt.figure()
        # plt.pcolormesh(f, c, spectrum, vmin=0,)
        # plt.pcolormesh(f, c, spectrum, cmap='jet', vmin=0)
        plt.contourf(f, c, spectrum, 20, vmin=0, cmap='jet')
        # plt.contourf(f, c, spectrum, 20, vmin=0)
        # plt.plot(data[:, 0], data[:, 1], 'r.', alpha=0.6)
        plt.plot(data[:, 0], data[:, 1], 'w.', alpha=0.8)
        # plt.plot(data[:, 0], data[:, 1], 'k.', alpha=0.8)
        plt.xlim([min(f), fmax])
        plt.ylim([cmin, cmax])
        plt.xlabel("Frequency(Hz)")
        plt.ylabel("Phase velocity(km/s)")
        plt.savefig("pick.jpg", dpi=600)
        plt.show()

    else:

        if plot_data:
            plt.figure()
            # plt.pcolormesh(f, c, spectrum, vmin=0,
            #                cmap='jet'
            #                )
            plt.contourf(f, c, spectrum, 20, vmin=0, cmap='jet')
            # plt.plot(c, spectrum[:, 500])
            plt.xlim([min(f), fmax])
            plt.ylim([min(c), cmax])
            plt.xlabel("Frequency(Hz)")
            plt.ylabel("Phase velocity(km/s)")
            plt.savefig("data.jpg", dpi=600)
            plt.show()
            sys.exit(0)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        print(f.shape, c.shape, spectrum.shape)
        obj_plot = ax.contourf(f, c, spectrum, 50, vmin=0, cmap='jet')
        # obj_plot = ax.pcolormesh(f, c, spectrum, cmap="jet", vmin=0)
        point = ax.plot([], [], 'k')
        ax.set_xlim([min(f), fmax])
        ax.set_ylim([cmin, cmax])
        ax.set_title("", fontsize=30)
        ax.set_xlabel("Frequency(Hz)")
        ax.set_ylabel("Phase velocity(km/s)")
        result = defaultdict(list)
        # pick = Pick(fig, ax, f, c, spectrum, width, result)
        pick = Pick(fig, ax, f, c, spectrum, width, nmax, result)
        pick.connect()
        plt.show()

        with open("data.txt", "w") as stream:
            for key, value in result.items():
                value = sorted(set(value))
                # for idx_f, idx_c in value:
                #     stream.write("{:14.7e}{:14.7e}{:5d}\n"
                #                  .format(f[idx_f], c[idx_c], key))
                for idx_f, c in value:
                    stream.write("{:14.7e}{:14.7e}{:5d}\n"
                                 .format(f[idx_f], c, key))



