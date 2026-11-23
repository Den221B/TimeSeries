"""Прогнозирование ряда продаж: модели, бэктест со скользящим началом и метрики (ДЗ 2)."""
import itertools
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox

sys.path.append(str(Path(__file__).resolve().parent.parent / "HW1"))
import retail_ts as rt  # noqa: E402  — функции подготовки ряда из ДЗ 1


# ---------- данные ----------

def load_series(code="85099B", end="2011-12-01"):
    """Ряды из ДЗ 1: день (6-дневный календарь), неделя и месяц (среднее за рабочий день).

    Пропуски (праздники, рождественская неделя) остаются NaN: модели получают интерполированную
    обучающую часть, а метрики считаются только по реально наблюдённым значениям.
    Неделя, обрезанная концом периода (28–30.11), отброшена.
    """
    df = rt.load_raw()
    cal = rt.store_calendar(df)
    cal = cal[cal.index < end]
    sales = rt.clean_sales(df)
    sales = sales[sales.date < end]
    daily = rt.build_daily(sales, code, cal)
    weekly = rt.aggregate(daily, "W-SUN").per_day
    weekly = weekly[weekly.index < pd.Timestamp(end)]
    monthly = rt.aggregate(daily, "MS").per_day
    return {"день": daily, "неделя": weekly, "месяц": monthly}


def check_index(series_dict):
    """Проверка индекса: порядок, дубликаты, шаги между наблюдениями, пропуски."""
    rows = {}
    for name, s in series_dict.items():
        steps = s.index.to_series().diff().dropna().dt.days.value_counts().sort_index()
        rows[name] = {"n": len(s), "начало": s.index[0].date(), "конец": s.index[-1].date(),
                      "по возрастанию": s.index.is_monotonic_increasing, "дубликаты": s.index.duplicated().sum(),
                      "шаги, дней (число)": dict(steps), "пропусков": int(s.isna().sum())}
    return pd.DataFrame(rows).T


# ---------- преобразование ----------

def to_bc(y):
    """Box–Cox для y + 1; λ оценивается только по переданной (обучающей) части."""
    z, lam = boxcox(np.asarray(y, float) + 1)
    return z, lam


def from_bc(z, lam):
    return np.clip(inv_boxcox(np.asarray(z), lam) - 1, 0, None)


# ---------- модели: f(y_train, h) -> прогноз длины h ----------

def naive(y, h, m=1):
    """Сезонный наивный прогноз: повторяем последние m значений (m=1 — обычный наивный)."""
    return np.resize(np.asarray(y)[-m:], h)


def ets(y, h, m=1):
    """Экспоненциальное сглаживание (Holt–Winters) на Box–Cox: затухающий тренд + сезонность при m > 1."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    z, lam = to_bc(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(z, trend="add", damped_trend=True,
                                   seasonal="add" if m > 1 else None,
                                   seasonal_periods=m if m > 1 else None,
                                   initialization_method="estimated").fit()
    return from_bc(fit.forecast(h), lam)


def select_arima(y, m=1, max_pq=2):
    """Подбор порядка (S)ARIMA по AIC на обучающей части.

    d = 0 и линейный тренд в модели: по ДЗ 1 ряд тренд-стационарен (единичного корня нет).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    z, _ = to_bc(y)
    seasonal = [(0, 0, 0, 0)] if m == 1 else [(P, 0, Q, m) for P in (0, 1) for Q in (0, 1)]
    best = (np.inf, None)
    for p, q in itertools.product(range(max_pq + 1), repeat=2):
        for so in seasonal:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    aic = SARIMAX(z, order=(p, 0, q), seasonal_order=so, trend="ct").fit(disp=False).aic
            except Exception:
                continue
            if aic < best[0]:
                best = (aic, ((p, 0, q), so))
    return best[1]


def arima(y, h, order, seasonal_order):
    """(S)ARIMA с фиксированным порядком на Box–Cox, линейный тренд."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    z, lam = to_bc(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(z, order=order, seasonal_order=seasonal_order, trend="ct").fit(disp=False)
    return from_bc(fit.forecast(h), lam)


def _features(z, t, lags, w, m):
    """Признаки для точки t: лаги 1..lags, среднее за последние w шагов, позиция в сезоне."""
    row = [z[t - k] for k in range(1, lags + 1)] + [z[t - w:t].mean()]
    if m > 1:
        row.append(t % m)
    return row


def lgbm(y, h, m=1, lags=12, w=12, min_leaf=10):
    """LightGBM на лагах ряда (Box–Cox), рекурсивный прогноз: каждый прогноз становится лагом для следующего.

    t % m — день недели (обучение всегда начинается с одной даты, поэтому позиция согласована).
    Ограничение: деревья не экстраполируют — прогноз остаётся в диапазоне обучающих значений, тренд не продлевается.
    """
    import lightgbm as lgb
    z, lam = to_bc(y)
    start = max(lags, w)
    X = [_features(z, t, lags, w, m) for t in range(start, len(z))]
    target = z[start:]
    params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 8,
              "min_data_in_leaf": min_leaf, "min_data_in_bin": 1, "bagging_fraction": 0.8,
              "bagging_freq": 1, "seed": 0, "verbose": -1}
    model = lgb.train(params, lgb.Dataset(np.array(X), np.array(target)), num_boost_round=300)
    hist = list(z)
    for _ in range(h):
        row = _features(np.array(hist), len(hist), lags, w, m)
        hist.append(model.predict(np.array([row]))[0])
    return from_bc(hist[-h:], lam)


# ---------- настройки эксперимента ----------

CONFIG = {
    # m — сезонный период (ДЗ 1: недельная сезонность только у дневного ряда, 6 рабочих дней)
    # n_origins/step — сколько начал прогноза и через сколько шагов; lags/w/min_leaf — для LightGBM
    "день":   dict(m=6, horizons=[1, 7, 31, 61], n_origins=8, step=6, lags=12, w=12, min_leaf=10, max_pq=2),
    "неделя": dict(m=1, horizons=[1, 4, 8], n_origins=8, step=1, lags=4, w=4, min_leaf=5, max_pq=2),
    "месяц":  dict(m=1, horizons=[1, 2], n_origins=2, step=1, lags=2, w=3, min_leaf=2, max_pq=1),
}


def make_origins(n, cfg):
    """Начала прогноза: последнее оставляет ровно max(h) шагов до конца ряда, остальные — раньше на step.

    Все горизонты оцениваются на одних и тех же началах — качество сравнимо между горизонтами.
    """
    last = n - max(cfg["horizons"])
    return [last - cfg["step"] * i for i in reversed(range(cfg["n_origins"]))]


def make_models(s, cfg, origins):
    """Набор моделей для частоты. Порядок ARIMA подбирается по AIC только на обучении первого начала.

    Имена одинаковые на всех частотах; при m > 1 «Наивный» — сезонный, «ARIMA» — SARIMA, ETS — с сезонностью.
    """
    m = cfg["m"]
    order = select_arima(s.iloc[:origins[0]].interpolate(limit_area="inside").bfill().values, m, cfg["max_pq"])
    models = {
        "Наивный": lambda y, h: naive(y, h, m),
        "ETS": lambda y, h: ets(y, h, m),
        "ARIMA": lambda y, h: arima(y, h, *order),
        "LightGBM": lambda y, h: lgbm(y, h, m, cfg["lags"], cfg["w"], cfg["min_leaf"]),
    }
    return models, order


def run_experiment(s, cfg):
    """Полный эксперимент для одной частоты: начала → модели → бэктест → метрики."""
    origins = make_origins(len(s), cfg)
    models, order = make_models(s, cfg, origins)
    fc = backtest(s, models, origins, max(cfg["horizons"]), cfg["m"])
    return fc, metrics(fc, cfg["horizons"]), order, origins


# ---------- бэктест ----------

def backtest(s, models, origins, max_h, m=1):
    """Прогноз со скользящим началом (expanding window).

    Для каждого начала o: обучение на s[:o] (только прошлое, пропуски интерполированы внутри),
    прогноз на min(max_h, len - o) шагов вперёд. Возвращает длинную таблицу прогнозов.
    scale — знаменатель MASE: средняя ошибка сезонного наивного прогноза на обучающей части.
    """
    rows = []
    for o in origins:
        train = s.iloc[:o].interpolate(limit_area="inside").bfill().values
        h = min(max_h, len(s) - o)
        scale = np.mean(np.abs(train[m:] - train[:-m]))
        for name, f in models.items():
            pred = f(train, h)
            rows.append(pd.DataFrame({"model": name, "origin": s.index[o], "step": np.arange(1, h + 1),
                                      "date": s.index[o:o + h], "y": s.values[o:o + h], "yhat": pred,
                                      "scale": scale}))
    return pd.concat(rows, ignore_index=True)


# ---------- метрики ----------

def metrics(fc, horizons):
    """Метрики для горизонта h — по всем шагам 1..h всех начал, где доступны все h шагов.

    MAE, RMSE — в единицах ряда; WAPE = Σ|e| / Σ|y| — доля ошибки от объёма продаж;
    MASE — ошибка относительно сезонного наивного прогноза на обучении (< 1 — лучше наивного).
    Наблюдения без факта (праздники) не учитываются.
    """
    out = []
    full = fc.groupby(["model", "origin"]).step.transform("max")
    for h in horizons:
        d = fc[(fc.step <= h) & (full >= h)].dropna(subset=["y"])
        e = d.y - d.yhat
        g = d.assign(ae=e.abs(), se=e ** 2, sae=e.abs() / d.scale).groupby("model")
        t = pd.DataFrame({"MAE": g.ae.mean(), "RMSE": np.sqrt(g.se.mean()),
                          "WAPE": g.ae.sum() / g.y.sum(), "MASE": g.sae.mean(),
                          "начал": g.origin.nunique(), "точек": g.size()})
        out.append(t.assign(h=h).reset_index())
    return pd.concat(out, ignore_index=True)


# ---------- графики ----------

COLORS = {"Наивный": "C0", "ETS": "C1", "ARIMA": "C2", "LightGBM": "C3"}  # одинаковые цвета на всех графиках


def plot_forecasts(s, fc, origin, title, history=60, ylabel="шт. в рабочий день"):
    """Факт и прогнозы всех моделей от одного начала."""
    fig, ax = plt.subplots(figsize=(15, 4.5))
    o = s.index.get_loc(origin)
    hist = s.iloc[max(0, o - history):]
    ax.plot(hist.index, hist.values, color="black", lw=1, marker=".", label="факт")
    ax.axvline(origin, color="grey", ls="--", lw=1, label="начало прогноза")
    for name, d in fc[fc.origin == origin].groupby("model", sort=False):
        ax.plot(d.date, d.yhat, lw=2, color=COLORS.get(name), label=name)
    ax.set(title=title, xlabel="дата", ylabel=ylabel)
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    return fig


def plot_metric_by_horizon(mt, metric="MASE"):
    """Метрика в зависимости от горизонта: по графику на частоту, линия на модель."""
    freqs = mt["частота"].unique()
    fig, axes = plt.subplots(1, len(freqs), figsize=(5.5 * len(freqs), 4))
    for ax, fr in zip(np.atleast_1d(axes), freqs):
        for name, d in mt[mt["частота"] == fr].groupby("model", sort=False):
            ax.plot(d.h.astype(str), d[metric], marker="o", color=COLORS.get(name), label=name)
        if metric == "MASE":
            ax.axhline(1, color="grey", ls=":", lw=1)
        ax.set(title=f"{fr}: {metric} по горизонтам", xlabel="горизонт, шагов", ylabel=metric)
        ax.legend()
    fig.tight_layout()
    return fig
