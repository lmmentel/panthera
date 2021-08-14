import argparse
import os
from .io import read_em_freq, read_pes, write_modes
from .plotting import plotmode_legacy


def plotmode_cli():
    """
    CLI interace for the plotting functions
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=int, help="number of the mode to be printed")
    parser.add_argument(
        "-s",
        "--sixth",
        default="em_freq",
        help='file with sixth order polynomial fit, default="em_freq"',
    )
    parser.add_argument(
        "-f",
        "--fourth",
        default="em_freq_4th",
        help='file with fourth order polynomial fit, default="em_freq_4th"',
    )
    parser.add_argument(
        "-p",
        "--pes",
        default="test_anharm",
        help='file with the potential energy surface (PES), default="test_anharm"',
    )
    parser.add_argument("-o", "--output", help="name of the output file")

    args = parser.parse_args()

    if os.path.exists(args.sixth):
        coeff6 = read_em_freq(args.sixth)
    else:
        raise OSError("File {} does not exist".format(args.sixth))
    if os.path.exists(args.fourth):
        coeff4 = read_em_freq(args.fourth)
    else:
        raise OSError("File {} does not exist".format(args.fourth))
    if os.path.exists(args.sixth):
        pes = read_pes(args.pes)
    else:
        raise OSError("File {} does not exist".format(args.pes))

    if args.mode > max(pes.keys()):
        raise ValueError(
            "Mode number {} unavailable, max mode number is: {}".format(
                args.mode, max(pes.keys())
            )
        )

    plotmode_legacy(args.mode, pes, coeff6, coeff4, args.output)


def write_modes_cli():
    """
    Parse the filename with multiple POSCARS form command line and write
    trajectory files with vibrational modes
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        default="POSCARs",
        help='name of the file with structures, default="POSCARs"',
    )
    parser.add_argument(
        "-d",
        "--dir",
        default="modes",
        help='directory to put the modes, default="modes"',
    )
    args = parser.parse_args()

    args.filename = os.path.abspath(args.filename)

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    os.chdir(args.dir)
    write_modes(args.filename)
