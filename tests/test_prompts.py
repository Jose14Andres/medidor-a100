import asyncio

from medidor.prompts import Calibracion, GeneradorPrompts, calibrar, dentro_de_tolerancia


def test_prefijo_unico_y_reproducible():
    a, b = GeneradorPrompts(7), GeneradorPrompts(7)
    pa = [a.nuevo(10) for _ in range(5)]
    assert pa == [b.nuevo(10) for _ in range(5)]
    assert [i for i, _ in pa] == [0, 1, 2, 3, 4]
    assert len({p for _, p in pa}) == 5                      # ningún prompt se repite
    assert GeneradorPrompts(8).nuevo(10) != pa[0]
    assert len(pa[0][1].split()) == 10 + 6                   # prefijo (2) + cuerpo + cola (4)


def _lineal(a, b, ruido=0):
    llamadas = []

    async def medir(palabras):
        llamadas.append(palabras)
        return round(a + b * palabras) + (ruido if len(llamadas) % 2 else 0)

    return medir, llamadas


def test_calibra_tokenizador_1_a_1():
    medir, _ = _lineal(6, 1.0)
    c = asyncio.run(calibrar(170, medir))
    assert c.ok and abs(c.tokens - 170) <= 1 and c.palabras == 164


def test_calibra_varios_tokens_por_palabra():
    medir, llamadas = _lineal(12, 1.17, ruido=1)
    for objetivo in (170, 615, 2390):
        c = asyncio.run(calibrar(objetivo, medir))
        assert c.ok, c.a_dict()
    assert len(llamadas) <= 3 * 8


def test_no_converge_si_el_servidor_trunca():
    async def truncado(palabras):
        return min(palabras + 6, 100)                       # num_ctx = 100

    c = asyncio.run(calibrar(170, truncado))
    assert not c.ok and c.tokens == 100 and len(c.sondeos) == 8


def test_tolerancia():
    assert dentro_de_tolerancia(173, 170) and not dentro_de_tolerancia(174, 170)
    assert Calibracion(100, 90, 102).ok and not Calibracion(100, 90, 103).ok


def test_objetivo_pequeno_exige_exactitud():
    # 20 tokens: 1 token de error ya es un 5 % (> 2 %); hay que seguir hasta el exacto.
    medir, _ = _lineal(6, 1.0)
    c = asyncio.run(calibrar(20, medir))
    assert c.ok and c.tokens == 20
