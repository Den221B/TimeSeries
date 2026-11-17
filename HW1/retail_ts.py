"""Функции анализа временного ряда продаж (датасет UCI Online Retail)."""
from pathlib import Path
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SERVICE_CODE = r"^\d{5}"  # обычные товары: код начинается с 5 цифр, остальное — POST, M, DOT, BANK CHARGES...


# ---------- загрузка ----------

def load_raw(data_dir=DATA_DIR):
    """Скачивает датасет при отсутствии, кэширует в pickle (xlsx читается ~40 с)."""
    data_dir = Path(data_dir)
    cache = data_dir / "online_retail.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    data_dir.mkdir(exist_ok=True)
    zpath = data_dir / "online_retail.zip"
    if not zpath.exists():
        urllib.request.urlretrieve(DATA_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        df = pd.read_excel(z.open("Online Retail.xlsx"), dtype={"InvoiceNo": str, "StockCode": str})
    df.to_pickle(cache)
    return df


# ---------- витрина: качество сырых данных ----------

def quality_report(df):
    """Таблица проблемных строк в сырых данных."""
    cancel = df.InvoiceNo.str.startswith("C")
    checks = {
        "Отмены (InvoiceNo начинается с C)": cancel,
        "Корректировки долга (InvoiceNo начинается с A)": df.InvoiceNo.str.startswith("A"),
        "Quantity < 0 без отмены (списания)": (df.Quantity < 0) & ~cancel,
        "UnitPrice <= 0": df.UnitPrice <= 0,
        "Служебные StockCode (POST, M, DOT, ...)": ~df.StockCode.str.match(SERVICE_CODE),
        "Пустой Description": df.Description.isna(),
        "Пустой CustomerID": df.CustomerID.isna(),
        "Полные дубликаты строк": df.duplicated(),
    }
    rep = pd.DataFrame({"строк": {k: int(v.sum()) for k, v in checks.items()}})
    rep["доля, %"] = (rep["строк"] / len(df) * 100).round(2)
    return rep


def store_calendar(df):
    """Все календарные дни периода и флаг «магазин работал» (был хоть один чек)."""
    days = df.InvoiceDate.dt.normalize()
    cal = pd.DataFrame(index=pd.date_range(days.min(), days.max(), name="date"))
    cal["is_open"] = cal.index.isin(days.unique())
    cal["weekday"] = cal.index.day_name()
    return cal


# ---------- витрина: очищенные продажи ----------

def drop_cancelled_orders(df, same_day=True):
    """Убирает продажи, у которых есть точная отмена (тот же клиент, товар и количество, отмена позже).

    same_day=True — только отмены в тот же день: это ошибки ввода (например, 80995 шт., отменённые
    через 12 минут). Возврат через недели спрос в день покупки не отменяет, поэтому его не трогаем.
    same_day=False — убираем продажу при любом возврате (ряд «что клиент оставил себе»).
    Возвращает (продажи без отменённых, число убранных строк).
    """
    sales = df[~df.InvoiceNo.str.startswith(("C", "A")) & (df.Quantity > 0)]
    canc = (df[df.InvoiceNo.str.startswith("C")].dropna(subset=["CustomerID"])  # без клиента не сопоставить
            .assign(Quantity=lambda d: -d.Quantity))
    key = ["CustomerID", "StockCode", "Quantity"]
    m = sales.dropna(subset=["CustomerID"]).reset_index().merge(
        canc[key + ["InvoiceDate"]], on=key, suffixes=("", "_c"))
    m = m[m.InvoiceDate <= m.InvoiceDate_c]
    if same_day:
        m = m[m.InvoiceDate.dt.normalize() == m.InvoiceDate_c.dt.normalize()]
    # ponytail: одна отмена может «погасить» несколько одинаковых продаж; на ряд влияет пренебрежимо
    bad = m.sort_values("InvoiceDate").groupby(key + ["InvoiceDate_c"])["index"].last().unique()
    return sales.drop(index=bad), len(bad)


def clean_sales(df, same_day=True):
    """Строки реальных продаж товаров: без отмен, списаний, служебных кодов и нулевых цен."""
    sales, _ = drop_cancelled_orders(df, same_day)
    sales = sales[(sales.UnitPrice > 0) & sales.StockCode.str.match(SERVICE_CODE)]
    return sales.assign(Revenue=sales.Quantity * sales.UnitPrice,
                        date=sales.InvoiceDate.dt.normalize())


def product_stats(sales, open_days):
    """Регулярность продаж каждого товара на рабочих днях магазина.

    open_days — DatetimeIndex рабочих дней; ноль в нерабочий день — не ноль спроса, поэтому их не берём.
    """
    daily = (sales.groupby(["date", "StockCode"]).Quantity.sum()
             .unstack(fill_value=0).reindex(open_days, fill_value=0))
    g = sales.groupby("StockCode")
    t = pd.DataFrame({
        "описание": g.Description.agg(lambda x: x.mode().iat[0]),
        "шт_всего": daily.sum(),
        "выручка": g.Revenue.sum().round(),
        "доля_дней_с_продажами": (daily > 0).mean().round(3),
        "месяцев": g.InvoiceDate.agg(lambda x: x.dt.to_period("M").nunique()),
        "клиентов": g.CustomerID.nunique(),
        "CV_дневной": (daily.std() / daily.mean()).round(2),
        "макс_день_доля": (daily.max() / daily.sum()).round(3),
        "цена_медиана": g.UnitPrice.median(),
    })
    return t.sort_values("доля_дней_с_продажами", ascending=False)


# ---------- временной ряд ----------

WEEKMASK = "Sun Mon Tue Wed Thu Fri"  # суббот в данных нет вообще — это не пропуск, а нерабочий день


def build_daily(sales, code, cal):
    """Дневной ряд продаж товара (шт.) на 6-дневном календаре магазина.

    0   — магазин работал, товар не купили (настоящий ноль спроса);
    NaN — праздник, магазин закрыт (наблюдения нет).
    """
    idx = pd.bdate_range(cal.index[0], cal.index[-1], freq="C", weekmask=WEEKMASK, name="date")
    q = sales.loc[sales.StockCode == code].groupby("date").Quantity.sum()
    return q.reindex(idx, fill_value=0).astype(float).where(cal.is_open.reindex(idx)).rename(code)


def aggregate(daily, freq):
    """Агрегация дневного ряда: сумма, число рабочих дней и среднее за рабочий день.

    Среднее за рабочий день сравнимо между периодами с разным числом рабочих дней
    (рождественская неделя, неделя с Пасхой). Неделя без рабочих дней → NaN.
    """
    r = daily.resample(freq)
    return pd.DataFrame({"sum": r.sum(min_count=1), "open_days": r.count(), "per_day": r.mean()})


def nan_runs(s):
    """Непрерывные отрезки пропусков: начало, конец, длина."""
    na = s.isna()
    grp = (na != na.shift()).cumsum()[na]
    return (pd.Series(s.index[na], index=grp.values).groupby(level=0)
            .agg(["first", "last", "size"]).set_axis(["начало", "конец", "длина"], axis=1)
            .reset_index(drop=True))


def zero_stats(s):
    """Доля нулей среди наблюдений и самая длинная серия нулей подряд."""
    obs = s.dropna()
    z = obs.eq(0)
    longest = int(z.groupby((~z).cumsum()).sum().max())
    return pd.Series({"наблюдений": len(obs), "нулей": int(z.sum()),
                      "доля нулей": round(z.mean(), 3), "макс. серия нулей": longest})


# ---------- выбросы ----------

def hampel_outliers(s, window=7, k=3.0):
    """Фильтр Хампеля: точка — выброс, если |x - скользящая медиана| > k * 1.4826 * MAD в окне.

    Медиана и MAD устойчивы к самим выбросам, а окно учитывает локальный уровень ряда (тренд).
    """
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    return (s - med).abs() > k * 1.4826 * mad.replace(0, mad[mad > 0].median())


def iqr_outliers(s, k=1.5):
    """Классическое правило IQR: выше Q3 + k*IQR или ниже Q1 - k*IQR."""
    q1, q3 = s.quantile([0.25, 0.75])
    return (s > q3 + k * (q3 - q1)) | (s < q1 - k * (q3 - q1))


# ---------- описательный анализ ----------

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]


def describe_series(series_dict):
    """Описательные статистики нескольких рядов (пропуски не учитываются)."""
    rows = {}
    for name, s in series_dict.items():
        s = s.dropna()
        rows[name] = {"n": len(s), "mean": s.mean(), "median": s.median(), "std": s.std(),
                      "CV": s.std() / s.mean(), "min": s.min(), "max": s.max(),
                      "skew": s.skew(), "kurtosis": s.kurt()}
    return pd.DataFrame(rows).round(2)


def level_spread(daily, freq="MS"):
    """Среднее и std дневных продаж внутри периода — меняется ли разброс вместе с уровнем."""
    t = daily.resample(freq).agg(["mean", "std"])
    t["CV"] = t["std"] / t["mean"]
    return t.round(2)


def weekday_kruskal(daily):
    """Тест Краскела–Уоллиса: одинаково ли распределены продажи по дням недели (H0 — да)."""
    from scipy.stats import kruskal
    s = daily.dropna()
    return kruskal(*[g.values for _, g in s.groupby(s.index.day_name())])


def order_profile(sales, code, freq="MS"):
    """Из чего складываются продажи: число чеков, клиентов и средний размер чека по периодам."""
    p = sales[sales.StockCode == code]
    t = p.groupby(pd.Grouper(key="date", freq=freq)).agg(
        чеков=("InvoiceNo", "nunique"), клиентов=("CustomerID", "nunique"), шт=("Quantity", "sum"))
    t["шт_на_чек"] = (t["шт"] / t["чеков"]).round(1)
    return t


# ---------- декомпозиция ----------

def stl_decompose(s, period=6, seasonal=31, trend=51, log=True):
    """Робастная STL. log=True — разложение log1p(ряда): мультипликативная модель становится аддитивной.

    seasonal=31 — длинное окно сглаживания сезонности: профиль недели почти постоянный во времени
    и не подстраивается под шум. trend=51 (~2 месяца рабочих дней) — тренд показывает долгосрочное
    направление, а не всплески оптовых заказов. Ряд должен быть без пропусков.
    """
    from statsmodels.tsa.seasonal import STL
    y = np.log1p(s) if log else s
    r = STL(y, period=period, seasonal=seasonal, trend=trend, robust=True).fit()
    return pd.DataFrame({"observed": r.observed, "trend": r.trend, "seasonal": r.seasonal,
                         "resid": r.resid, "weight": r.weights}, index=s.index)


def strength(dec):
    """Сила тренда и сезонности (Hyndman): 1 - Var(R)/Var(T+R) и 1 - Var(R)/Var(S+R), от 0 до 1."""
    r = dec.resid.var()
    return pd.Series({"сила тренда F_T": max(0, 1 - r / (dec.trend + dec.resid).var()),
                      "сила сезонности F_S": max(0, 1 - r / (dec.seasonal + dec.resid).var())}).round(3)


def seasonality_permutation_test(s, n=200, seed=0, **stl_kw):
    """Насколько F_S выше, чем у того же ряда с перемешанными днями (сезонности там нет по построению).

    p-value — доля перестановок, где F_S не меньше реального.
    """
    rng = np.random.default_rng(seed)
    fs = strength(stl_decompose(s, **stl_kw))["сила сезонности F_S"]
    null = np.array([strength(stl_decompose(pd.Series(rng.permutation(s.values), index=s.index), **stl_kw))
                     ["сила сезонности F_S"] for _ in range(n)])
    return pd.Series({"F_S": fs, "F_S перемешанных, 95%": np.quantile(null, 0.95).round(3),
                      "p-value": (null >= fs).mean()})


def seasonal_factors(dec, split=None):
    """Сезонные коэффициенты по дням недели: exp(среднего сезонного компонента в лог-шкале).

    1.5 — день продаёт в 1.5 раза больше типичного. split — дата, чтобы сравнить профиль до и после.
    """
    s = dec.seasonal
    by = [s.index.day_name()] if split is None else [np.where(s.index < split, f"до {split}", f"после {split}"),
                                                       s.index.day_name()]
    f = np.exp(s.groupby(by).mean())
    return (f if split is None else f.unstack(0)).reindex(WEEKDAYS).round(2)


# ---------- стационарность ----------

def _verdict(adf_c, adf_ct, kpss_c, kpss_ct, alpha=0.05):
    if adf_c < alpha and kpss_c >= alpha:
        return "стационарен"
    if adf_ct < alpha and kpss_ct >= alpha:
        return "тренд-стационарен"
    if adf_ct >= alpha and kpss_c < alpha:
        return "единичный корень"
    return "неоднозначно"


def stationarity_tests(series_dict):
    """ADF и KPSS для нескольких рядов + вывод.

    ADF:  H0 — единичный корень (нестационарен); p < 0.05 → отвергаем.
    KPSS: H0 — стационарен (вокруг константы 'c' или тренда 'ct'); p < 0.05 → отвергаем.
    p-value KPSS табличный и обрезан в [0.01, 0.1]: 0.1 означает «≥ 0.1».
    acf1 — автокорреляция на лаге 1; около -0.5 — признак передифференцирования.
    """
    import warnings
    from statsmodels.tsa.stattools import acf, adfuller, kpss
    rows = {}
    for name, s in series_dict.items():
        s = s.dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = {"n": len(s),
                 "ADF p (c)": adfuller(s, regression="c", autolag="AIC")[1],
                 "ADF p (ct)": adfuller(s, regression="ct", autolag="AIC")[1],
                 "KPSS p (c)": kpss(s, regression="c", nlags="auto")[1],
                 "KPSS p (ct)": kpss(s, regression="ct", nlags="auto")[1],
                 "acf1": acf(s, nlags=1)[1]}
        r["вывод"] = _verdict(r["ADF p (c)"], r["ADF p (ct)"], r["KPSS p (c)"], r["KPSS p (ct)"])
        rows[name] = r
    return pd.DataFrame(rows).T.round(3)


def variance_check(series_dict, freq="MS"):
    """Стабильность дисперсии: корреляция среднего и std по периодам, отношение max/min std, скошенность."""
    rows = {}
    for name, s in series_dict.items():
        g = s.resample(freq).agg(["mean", "std"])
        rows[name] = {"corr(mean, std)": g["mean"].corr(g["std"]),
                      "std max/min": g["std"].max() / g["std"].min(), "skew": s.skew()}
    return pd.DataFrame(rows).T.round(2)


def boxcox(s):
    """Box–Cox для y + 1 (в ряде есть нули); λ подбирается методом максимального правдоподобия."""
    from scipy.stats import boxcox as _bc
    y, lam = _bc(s + 1)
    return pd.Series(y, index=s.index, name=s.name), lam


def detrend_linear(s):
    """Вычитает линейный тренд (МНК по номеру наблюдения). Возвращает (остаток, наклон на шаг)."""
    x = np.arange(len(s))
    slope, intercept = np.polyfit(x, s.values, 1)
    return s - (intercept + slope * x), slope


# ---------- графики ----------

def plot_decomposition(dec, title, log=True):
    """Наблюдения, тренд, сезонность и остаток STL; тренд дополнительно в штуках."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    unit = "log(1 + шт.)" if log else "шт."
    axes[0].plot(dec.index, dec.observed, lw=0.7, label="ряд")
    axes[0].plot(dec.index, dec.trend, lw=2, label="тренд")
    axes[1].plot(dec.index, np.expm1(dec.trend) if log else dec.trend, lw=2, color="C1", label="тренд")
    axes[2].plot(dec.index, dec.seasonal, lw=0.7, color="C2", label="сезонность (период 6)")
    axes[3].scatter(dec.index, dec.resid, s=8, c=np.where(dec.weight < 0.5, "red", "C0"))
    axes[3].scatter([], [], c="red", s=8, label="остаток с низким весом (выброс для STL)")
    axes[3].axhline(0, color="black", lw=0.5)
    for ax, t, yl in zip(axes, ["Ряд и тренд", "Тренд в штуках", "Сезонная компонента", "Остаток"],
                         [unit, "шт. в день", unit, unit]):
        ax.set(title=t, ylabel=yl)
        ax.legend(loc="upper left")
    axes[3].set_xlabel("дата")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


def plot_rolling(s, window, title, ylabel="шт."):
    """Ряд, скользящее среднее и скользящее std — изменение уровня и разброса во времени."""
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(s.index, s.values, lw=0.6, alpha=0.5, label="ряд")
    ax.plot(s.index, s.rolling(window, center=True).mean(), lw=2, label=f"скользящее среднее ({window})")
    ax.plot(s.index, s.rolling(window, center=True).std(), lw=2, label=f"скользящее std ({window})")
    ax.set(title=title, xlabel="дата", ylabel=ylabel)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_weekday_profile(daily, title):
    """Распределение дневных продаж по дням недели: boxplot + среднее."""
    s = daily.dropna()
    groups = [s[s.index.day_name() == d].values for d in WEEKDAYS]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(groups, tick_labels=[d[:3] for d in WEEKDAYS], showfliers=False)
    ax.plot(range(1, len(WEEKDAYS) + 1), [g.mean() for g in groups], "o", color="red", label="среднее")
    ax.set(title=title, xlabel="день недели", ylabel="шт. в день")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_acf_pacf(s, lags, title):
    """ACF и PACF ряда (без пропусков)."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 3.5))
    plot_acf(s, lags=lags, ax=a1, title=f"ACF: {title}")
    plot_pacf(s, lags=lags, ax=a2, title=f"PACF: {title}", method="ywm")
    for a in (a1, a2):
        a.set(xlabel="лаг", ylabel="корреляция")
    fig.tight_layout()
    return fig


def plot_series(series_dict, outliers=None, title="", ylabel="шт."):
    """Ряды друг под другом; outliers — словарь {имя: булева маска} для отметки выбросов."""
    fig, axes = plt.subplots(len(series_dict), 1, figsize=(15, 3.2 * len(series_dict)), squeeze=False)
    for ax, (name, s) in zip(axes[:, 0], series_dict.items()):
        ax.plot(s.index, s.values, marker=".", lw=0.8, label=name)
        if outliers is not None and name in outliers:
            o = s[outliers[name]]
            ax.scatter(o.index, o.values, color="red", zorder=3, label=f"выбросы ({len(o)})")
        ax.set(title=name, xlabel="дата", ylabel=ylabel)
        ax.legend(loc="upper left")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


def plot_store_activity(df, cal):
    """Число чеков по дням недели и по датам; нерабочие дни (кроме суббот) отмечены."""
    inv = df.drop_duplicates("InvoiceNo")
    per_day = inv.groupby(inv.InvoiceDate.dt.normalize()).size().reindex(cal.index, fill_value=0)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_wd = inv.InvoiceDate.dt.day_name().value_counts().reindex(order, fill_value=0)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 3]})
    a1.bar(by_wd.index.str[:3], by_wd.values)
    a1.set(title="Чеки по дням недели", xlabel="день недели", ylabel="число чеков")
    a2.plot(per_day.index, per_day.values, lw=0.8, label="чеков за день")
    closed = cal[~cal.is_open & (cal.weekday != "Saturday")].index
    a2.scatter(closed, [0] * len(closed), color="red", zorder=3, label="закрыто (не суббота)")
    a2.set(title="Число чеков по датам", xlabel="дата", ylabel="число чеков")
    a2.legend()
    fig.tight_layout()
    return fig


def plot_monthly_grid(sales, codes, ncols=3):
    """Помесячные продажи (шт.) нескольких товаров — для выбора кандидата."""
    m = (sales[sales.StockCode.isin(codes)]
         .groupby([pd.Grouper(key="date", freq="MS"), "StockCode"]).Quantity.sum().unstack())
    nrows = -(-len(codes) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True)
    for ax, code in zip(axes.flat, codes):
        ax.bar(m.index, m[code], width=20)
        desc = sales.loc[sales.StockCode == code, "Description"].mode().iat[0]
        ax.set_title(f"{code}: {desc[:30]}", fontsize=10)
        ax.set_ylabel("шт. в месяц")
        ax.tick_params(axis="x", rotation=45)
    for ax in axes.flat[len(codes):]:
        ax.remove()
    fig.suptitle("Помесячные продажи кандидатов", y=1.0)
    fig.tight_layout()
    return fig
