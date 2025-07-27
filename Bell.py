#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 11:15:45 2025

@author: anna-elisamaassen
"""


import io
import streamlit as st
import segno
import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr
from pyngrok import conf, ngrok

# 1) Dein ngrok-Authtoken (von https://dashboard.ngrok.com)
conf.get_default().auth_token = "30SHptRr8OVf24tIcPdfmat3hOW_89cqp3Q9WEvyfPqVnGWAL"

# 2) Tunnel EINMAL pro Session öffnen (kostenfrei → keine subdomain-Angabe!)
if "public_url" not in st.session_state:
    # bind_tls=True liefert Dir eine HTTPS-URL
    tunnel = ngrok.connect(addr=8501, bind_tls=True)
    st.session_state.public_url = tunnel.public_url

public_url = st.session_state.public_url

# 3) QR-Code generieren
buf = io.BytesIO()
qr = segno.make(public_url)
qr.save(buf, kind="png", scale=5)
buf.seek(0)

# 4) Streamlit-Seite
st.set_page_config(layout="wide")
st.title("Bell-Test: Blendenöffnungs-Auswertung")

st.markdown(
    f"**Scanne diesen QR-Code, um die App von überall zu öffnen:**  \n"
    f"{public_url}"
)
st.image(buf, width=200)

st.markdown("""
Gib hier Deine Messdaten ein.  
- D: Durchmesser der Blendenöffnung in mm  
- S: gemessene S-Werte  
- s_std: Fehler der S-Werte  
- d_std: Fehler auf D (konstant)  
""")

col1, col2 = st.columns(2)
with col1:
    D_str     = st.text_area("D [mm]",      "5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5")
    s_std_str = st.text_area("Fehler s_std","0.065,0.021,0.011,0.011,0.009,0.009,0.007,0.008,0.007,0.006,0.007")
with col2:
    S_str = st.text_area("S-Werte", "2.520,2.518,2.494,2.499,2.471,2.451,2.459,2.453,2.455,2.449,2.450")
    d_std = st.number_input("Fehler d_std (konstant)", value=0.1, format="%.3f")

if st.button("Auswerten"):
    try:
        # Parsen
        D     = np.array([float(x) for x in D_str.split(",") if x.strip()])
        S     = np.array([float(x) for x in S_str.split(",") if x.strip()])
        s_std = np.array([float(x) for x in s_std_str.split(",") if x.strip()])

        # ODR-Fit
        def lin(beta, x): return beta[0]*x + beta[1]
        model   = odr.Model(lin)
        data    = odr.RealData(D, S, sx=d_std, sy=s_std)
        odr_obj = odr.ODR(data, model, beta0=[1.0, 0.0])
        out     = odr_obj.run()
        m, b    = out.beta
        dm, db  = out.sd_beta

        # Ergebnis & Plot
        x_fit    = np.linspace(D.min()-1, D.max()+1, 200)
        y_fit    = lin([m, b], x_fit)
        residuen = S - lin([m, b], D)
        chi2     = np.sum(residuen**2 / ((D*dm)**2 + db**2))

        st.subheader("Fit-Ergebnisse")
        st.write(f"Steigung m = {m:.4f} ± {dm:.4f}")
        st.write(f"Achsenabschnitt b = {b:.4f} ± {db:.4f}")
        st.write(f"Reduced χ² = {chi2/(len(S)-2):.2f}")

        fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex=True)
        ax[0].errorbar(D, S, xerr=d_std, yerr=s_std, fmt='o', label='Messdaten')
        ax[0].plot(x_fit, y_fit, 'r-', label='ODR-Fit')
        ax[0].axhline(2*np.sqrt(2), color='C1', ls='--', label="Tsirelson")
        ax[0].axhline(2.0,             color='C2', ls='--', label="klass. Limit")
        ax[0].set_ylabel('S-Wert'); ax[0].legend(); ax[0].grid(True)

        ax[1].errorbar(D, residuen, xerr=d_std, yerr=s_std, fmt='o')
        ax[1].axhline(0, color='gray', ls='--')
        ax[1].set_xlabel('D [mm]'); ax[1].set_ylabel('Residuen'); ax[1].grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Fehler beim Auswerten: {e}")