# ✈️ FAAN Airline Delay-Predict

An end-to-end machine learning and web platform predicting flight delays and powering a complete airline booking experience for Akanu Ibiam International Airport (ENU).

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Backend-Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ML-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

---

> **Built for Real-World Domestic Aviation**  
> Designed around real Nigerian domestic flight operations (Air Peace, Ibom Air, United Nigeria, Enugu Air) operating out of Enugu (ENU).

---

## 🏗️ System Architecture

```text
┌─────────────────────────┐       ┌───────────────────────────┐       ┌────────────────────────┐
│     Data Layer          │  ───> │     Machine Learning      │  ───> │    REST API Layer      │
│  enu_flight_delays.csv  │       │ Classification + Regress. │       │      (Flask App)       │
└─────────────────────────┘       └───────────────────────────┘       └────────────────────────┘
                                                                                  │
                                                                                  ▼
                                                                      ┌────────────────────────┐
                                                                      │     User Interface     │
                                                                      │  Dashboard & Booking   │
                                                                      └────────────────────────┘

✨ Why This Project Stands Out
Dual-Model ML Pipeline: Features combined classification (delay risk) and regression (delay duration in minutes) models working simultaneously.

End-to-End Integration: Seamless pipeline from raw CSV data to live model inference and dynamic UI presentation.

Scalable Architecture: Modular design (Data → ML → API → UI) easily adapted to other airports beyond ENU.

Full Product Features: Complete user session management, persistent flight booking, and dynamic multi-airline dashboards.

Localized Context: Built for domestic Nigerian aviation routes and airlines.





