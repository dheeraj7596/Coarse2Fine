import matplotlib.pyplot as plt


def compute_lambda(step, max_steps):
    temp = 1 - step / max_steps
    if temp < 0:
        return 0
    else:
        return temp


if __name__ == "__main__":
    f = open("/Users/dheerajmekala/Work/Coarse2Fine/output/gpt2_fine_nopadloss_linear_reg_arts.txt", "r")
    lines = f.readlines()
    f.close()

    kl_loss = []
    lambdas = []
    ce_loss = []
    global_step = 0
    epochs = 5

    kl_str = "KL-loss"
    ce_str = "CE-loss"
    for line in lines:
        line = line.strip()
        if kl_str and ce_str in line:
            toks = line.split(kl_str)
            if len(toks) == 1:
                continue
            temp = toks[1]
            vals = temp.strip().split(ce_str)
            ce = vals[1].strip()
            if float(ce) == 0:
                continue
            kl = vals[0].strip()
            ce_loss.append(float(ce))
            kl_loss.append(float(kl))
            lambdas.append(compute_lambda(global_step, epochs * 1000))
            global_step += 1

    its = list(range(len(kl_loss)))[10:]

    a = plt.plot(its, kl_loss[10:], label="KL-Div")
    b = plt.plot(its, ce_loss[10:], label="CE-loss")
    c = plt.plot(its, lambdas[10:], label="lambda")
    plt.xlabel("#steps", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.legend(title='title', bbox_to_anchor=(0.95, 1), loc='upper left')
    plt.legend(prop={'size': 9}, loc="upper right")

    plt.title("Arts")

    plt.savefig('/Users/dheerajmekala/Desktop/losses.png')
    # plt.show()
