# 📦 Supply Chain Bottleneck & RL Recommendations

An interactive **Streamlit web app** for identifying bottlenecks in
supply chain data and simulating reinforcement learning (RL)--based
decision recommendations. The app analyzes order data, visualizes
performance, and suggests optimal mitigation strategies such as shipping
upgrades or fraud reviews.

------------------------------------------------------------------------

## 🚀 Features

-   **Dataset Upload**\
    Upload your own CSV (e.g., `DataCoSupplyChainDataset.csv`) to
    perform dynamic analysis.

-   **Customizable Analysis Settings**

    -   Adjustable thresholds for late deliveries, cancellations, fraud,
        and revenue loss.\
    -   Tunable RL hyperparameters for simulation: learning rate,
        discount factor, epsilon decay, etc.

-   **Bottleneck Identification**\
    Detect categories with poor delivery, cancellation, or fraud
    performance.

-   **Reinforcement Learning Simulation**

    -   Implements a basic Q-learning algorithm to optimize
        shipping/fraud decisions.\
    -   Visualizes agent learning progress using reward and epsilon
        decay plots.

-   **AI-Powered Recommendations**\
    Provides actionable insights on which categories require
    interventions.

------------------------------------------------------------------------

## 🧩 Requirements

### Install dependencies:

``` bash
pip install streamlit pandas numpy matplotlib
```

------------------------------------------------------------------------

## 🗂️ File Structure

    .
    ├── 26305973-2575-4d75-8ca0-a9f1bb4059f0.py   # Main Streamlit application
    ├── DataCoSupplyChainDataset.csv              # (Optional) Example dataset
    └── README.md                                 # Project documentation

------------------------------------------------------------------------

## 🧠 How It Works

1.  **Load Dataset**\
    The app expects columns such as:
    -   `Category Name`
    -   `Order Id`
    -   `Order Status`
    -   `Late_delivery_risk`
    -   `Sales`
    -   `Delivery Status`
    -   `Days for shipment (scheduled)`
2.  **Analyze Bottlenecks**\
    Computes per-category metrics:
    -   Late delivery rate (%)
    -   Cancellation rate (%)
    -   Fraud rate (%)
    -   Average lead time and order value
    -   Revenue loss due to late deliveries
3.  **Run Reinforcement Learning**\
    Simulates an agent improving decisions based on order outcomes:
    -   Actions: *Keep Standard Shipping*, *Upgrade to First Class*,
        *Flag for Fraud Review*
    -   Rewards are updated using the **Bellman equation** to refine
        future policy.
4.  **Generate Recommendations**\
    Flags categories exceeding threshold values and provides AI-driven
    mitigation strategies.

------------------------------------------------------------------------

## ▶️ Usage

Run the app locally:

``` bash
streamlit run 26305973-2575-4d75-8ca0-a9f1bb4059f0.py
```

Upload your dataset when prompted and adjust thresholds or
hyperparameters using the sidebar.

------------------------------------------------------------------------

## 📊 Output

-   **Interactive data tables** summarizing bottleneck metrics.\
-   **Learning progress plots** showing:
    -   Epsilon decay (exploration rate)
    -   Moving average of rewards\
-   **Recommendations table** suggesting targeted actions per category.

------------------------------------------------------------------------

## 🧪 Example Insights

-   Categories with \>15% late deliveries or \>5% cancellations are
    flagged.\
-   RL agent learns optimal actions that balance customer satisfaction
    and revenue.\
-   Results can guide process redesign, supplier selection, or logistics
    upgrades.

------------------------------------------------------------------------

## 🧩 Technologies Used

-   **Python 3.10+**
-   **Streamlit** -- Interactive web UI\
-   **Pandas** -- Data manipulation\
-   **NumPy** -- Numerical processing\
-   **Matplotlib** -- Visualization

------------------------------------------------------------------------

## 🧠 Future Enhancements

-   Integrate real-time supply chain dashboards.\
-   Deploy using Docker or Streamlit Cloud.\
-   Use advanced RL algorithms (DQN, PPO).\
-   Add dynamic what-if scenario analysis.
