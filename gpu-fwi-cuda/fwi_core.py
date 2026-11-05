import math

# Day-length factors (Van Wagner 1987), index = month-1
EL = (6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0)
FL = (-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6)


def ffmc_step(ffmc0, T, H, W, rain):
    mo = 147.2 * (101.0 - ffmc0) / (59.5 + ffmc0)
    if rain > 0.5:
        rf = rain - 0.5
        if mo > 150.0:
            mo = (mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
                  + 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf))
        else:
            mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        if mo > 250.0:
            mo = 250.0
    ed = 0.942 * H ** 0.679 + 11.0 * math.exp((H - 100.0) / 10.0) + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H))
    if mo > ed:
        ko = 0.424 * (1.0 - (H / 100.0) ** 1.7) + 0.0694 * math.sqrt(W) * (1.0 - (H / 100.0) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * T)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    else:
        ew = 0.618 * H ** 0.753 + 10.0 * math.exp((H - 100.0) / 10.0) + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H))
        if mo < ew:
            k1 = 0.424 * (1.0 - ((100.0 - H) / 100.0) ** 1.7) + 0.0694 * math.sqrt(W) * (1.0 - ((100.0 - H) / 100.0) ** 8)
            kw = k1 * 0.581 * math.exp(0.0365 * T)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo
    f = 59.5 * (250.0 - m) / (147.2 + m)
    return min(max(f, 0.0), 101.0)


def dmc_step(dmc0, T, H, rain, el):
    t = max(T, -1.1)
    rk = 1.894 * (t + 1.1) * (100.0 - H) * el * 1e-4
    if rain > 1.5:
        rw = 0.92 * rain - 1.27
        wmi = 20.0 + 280.0 / math.exp(0.023 * dmc0)
        if dmc0 <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc0)
        elif dmc0 <= 65.0:
            b = 14.0 - 1.3 * math.log(dmc0)
        else:
            b = 6.2 * math.log(dmc0) - 17.2
        wmr = wmi + 1000.0 * rw / (48.77 + b * rw)
        pr = 43.43 * (5.6348 - math.log(wmr - 20.0))
    else:
        pr = dmc0
    pr = max(pr, 0.0)
    return max(pr + rk, 0.0)


def dc_step(dc0, T, rain, fl):
    t = max(T, -2.8)
    pe = max((0.36 * (t + 2.8) + fl) / 2.0, 0.0)
    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        qo = 800.0 * math.exp(-dc0 / 400.0)
        qr = qo + 3.937 * rd
        dr = max(400.0 * math.log(800.0 / qr), 0.0)
        return dr + pe
    return dc0 + pe


def isi_calc(ffmc, W):
    mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * math.exp(-0.1386 * mo) * (1.0 + mo ** 5.31 / 4.93e7)
    return 0.208 * math.exp(0.05039 * W) * ff


def bui_calc(dmc, dc):
    if dmc == 0.0 and dc == 0.0:
        return 0.0
    if dmc <= 0.4 * dc:
        b = 0.8 * dmc * dc / (dmc + 0.4 * dc)
    else:
        b = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
    return max(b, 0.0)


def fwi_calc(isi, bui):
    if bui <= 80.0:
        bb = 0.1 * isi * (0.626 * bui ** 0.809 + 2.0)
    else:
        bb = 0.1 * isi * (1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui)))
    if bb <= 1.0:
        return bb
    return math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647)
