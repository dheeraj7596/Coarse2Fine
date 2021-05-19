import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    x = [0, np.log(2), np.log(3), np.log(5), np.log(7)]

    micro = [0.7782, 0.7725, 0.785, 0.7795, 0.7871]
    macro = [0.7782, 0.7723, 0.7847, 0.7800, 0.7870]

    fig, ax = plt.subplots()

    ax.plot(x, micro, label="Micro F1")
    ax.plot(x, macro, label="Macro F1")

    exp = lambda x: np.exp(x)
    log = lambda x: np.log(x)

    ax.set_xscale('function', functions=(exp, log))
    labels = ["0", "log2", "log3", "log4", "log5", "log6", "log7"]
    ax.set_xticks([0, np.log(2), np.log(3), np.log(4), np.log(5), np.log(6), np.log(7)])
    ax.set_yticks([0.76, 0.77, 0.78, 0.79])
    ax.set_xticklabels(labels)


    plt.xlabel(r'Epsilon ($\epsilon$)')
    plt.ylabel('F1-score')
    plt.legend(prop={'size': 12}, loc="lower right")

    # plt.show()
    plt.savefig("epsilon2.pdf")