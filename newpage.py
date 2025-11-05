import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the page configuration for a better layout
st.set_page_config(page_title="Supply Chain Bottleneck Analysis", layout="wide")

# --- Title ---
st.title("📦 Supply Chain Bottleneck & RL Recommendations")

# --- Load Dataset ---
# Use a file uploader for more flexibility, or keep the hardcoded path
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="latin1")
        st.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()
else:
    st.info("Please upload the 'DataCoSupplyChainDataset.csv' file to begin analysis.")
    st.stop()


# --- Sidebar Settings ---
st.sidebar.header("⚙️ Analysis Settings")

# RL Hyperparameters
st.sidebar.subheader("🤖 RL Hyperparameters")
learning_rate = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1, 0.01, help="How quickly the agent learns from new information.")
discount_factor = st.sidebar.slider("Discount Factor (γ)", 0.1, 1.0, 0.9, 0.05, help="How much the agent values future rewards over immediate rewards.")
epsilon_start = st.sidebar.slider("Starting Epsilon (ε)", 0.1, 1.0, 1.0, 0.05, help="Initial probability of the agent exploring random actions.")
min_epsilon = st.sidebar.slider("Minimum Epsilon", 0.01, 0.5, 0.05, 0.01, help="The lowest exploration rate the agent will reach.")
decay_rate = st.sidebar.slider("Epsilon Decay Rate", 0.900, 0.9999, 0.999, 0.0001, format="%.4f", help="Rate at which the exploration probability decreases.")

# Bottleneck Thresholds
st.sidebar.header("📊 Bottleneck Thresholds")
late_threshold = st.sidebar.slider("Late Delivery Rate %", 5, 50, 15, 1, help="Mark categories with late deliveries above this percentage as bottlenecks.")
cancel_threshold = st.sidebar.slider("Cancellation Rate %", 1, 20, 5, 1, help="Mark categories with cancellations above this percentage as bottlenecks.")
fraud_threshold = st.sidebar.slider("Fraud Rate %", 1, 10, 2, 1, help="Mark categories with suspected fraud above this percentage as bottlenecks.")
loss_threshold = st.sidebar.number_input("Revenue Loss ($)", min_value=1000, value=5000, step=500, help="Mark categories with revenue loss from late deliveries above this amount as bottlenecks.")

# --- Analysis Section ---
st.subheader("🔍 Bottleneck Analysis by Category")

# Use a function to avoid re-running complex logic on every interaction
@st.cache_data
def run_analysis(dataf):
    category_group = dataf.groupby("Category Name")
    
    analysis = category_group.agg(
        total_orders=("Order Id", "nunique"),
        late_delivery_rate=("Late_delivery_risk", "mean"),
        avg_lead_time=("Days for shipment (scheduled)", "mean"),
        avg_order_value=("Sales", "mean")
    ).reset_index()

    def get_status_rate(group, status):
        return (group["Order Status"] == status).sum() / len(group) if len(group) > 0 else 0

    cancellation_rate = category_group.apply(lambda x: get_status_rate(x, "CANCELED")).rename("cancellation_rate").reset_index()
    fraud_rate = category_group.apply(lambda x: get_status_rate(x, "SUSPECTED_FRAUD")).rename("fraud_rate").reset_index()

    analysis = pd.merge(analysis, cancellation_rate, on="Category Name")
    analysis = pd.merge(analysis, fraud_rate, on="Category Name")

    analysis["late_delivery_rate"] *= 100
    analysis["cancellation_rate"] *= 100
    analysis["fraud_rate"] *= 100

    dataf["Revenue_Loss"] = dataf["Late_delivery_risk"] * dataf["Sales"]
    category_loss = dataf.groupby("Category Name")["Revenue_Loss"].sum().reset_index()
    analysis = pd.merge(analysis, category_loss, on="Category Name")
    return analysis

category_analysis = run_analysis(df.copy()) # Use a copy to prevent caching issues

st.dataframe(category_analysis.style.format({
    'late_delivery_rate': '{:.2f}%',
    'cancellation_rate': '{:.2f}%',
    'fraud_rate': '{:.2f}%',
    'avg_lead_time': '{:.2f}',
    'avg_order_value': '${:,.2f}',
    'Revenue_Loss': '${:,.2f}'
}))


# --- Reinforcement Learning Simulation ---
st.subheader("🤖 Reinforcement Learning Simulation")

@st.cache_data
def run_rl_simulation(_df, lr, gamma, eps_start, eps_min, decay):
    states = _df["Category Name"].unique()
    actions = ["Keep Standard Shipping", "Upgrade to First Class", "Flag for Fraud Review"]
    q_table = pd.DataFrame(0, index=states, columns=actions, dtype=np.float32)

    epsilon = eps_start
    rewards_log, epsilon_log = [], []

    # Use a smaller sample for faster execution in the app
    orders = _df.sample(min(len(_df), 5000), random_state=42).iterrows()

    for step, (index, order) in enumerate(orders):
        state = order["Category Name"]
        if np.random.rand() < epsilon:
            action_taken = np.random.choice(actions) # Explore
        else:
            action_taken = q_table.loc[state].idxmax() # Exploit

        # Calculate reward
        reward = 0
        if order["Delivery Status"] == "Late delivery":
            reward -= 20
        elif order["Delivery Status"] in ["Advance shipping", "Shipping on time"]:
            reward += 10
        if order["Order Status"] == "SUSPECTED_FRAUD":
            reward += 50 if action_taken == "Flag for Fraud Review" else -50

        # Q-table update using the Bellman equation
        old_value = q_table.loc[state, action_taken]
        next_max = q_table.loc[state].max()
        new_value = old_value + lr * (reward + gamma * next_max - old_value)
        q_table.loc[state, action_taken] = new_value

        # Log and decay epsilon
        rewards_log.append(reward)
        epsilon_log.append(epsilon)
        epsilon = max(eps_min, epsilon * decay)
    
    return q_table, rewards_log, epsilon_log

q_table, rewards_log, epsilon_log = run_rl_simulation(df, learning_rate, discount_factor, epsilon_start, min_epsilon, decay_rate)

# --- Plot RL Learning Curves ---
st.subheader("📈 Learning Progress")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Epsilon Decay Plot
ax[0].plot(epsilon_log, label="Epsilon (Exploration Rate)")
ax[0].set_title("Exploration vs. Exploitation Trade-off")
ax[0].set_xlabel("Training Steps (Orders Processed)")
ax[0].set_ylabel("Epsilon")
ax[0].grid(True, linestyle='--', alpha=0.6)
ax[0].legend()

# Reward Trend Plot
moving_avg = pd.Series(rewards_log).rolling(window=200, min_periods=1).mean()
ax[1].plot(moving_avg, color="green", label=f"Moving Avg Reward (window=200)")
ax[1].set_title("Reward Trend (Agent Performance)")
ax[1].set_xlabel("Training Steps (Orders Processed)")
ax[1].set_ylabel("Average Reward")
ax[1].grid(True, linestyle='--', alpha=0.6)
ax[1].legend()

plt.tight_layout()
st.pyplot(fig)

# --- Current Data Situation 
# --- Late Delivery Rate Over Time ---
st.subheader("📉 Late Delivery Rate Over Time")

# Ensure 'Order Date' is parsed as datetime
df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")

# Group by month to calculate late delivery rate trend
late_rate_trend = df.groupby(df["order_date"].dt.to_period("M"))["Late_delivery_risk"].mean().reset_index()
late_rate_trend["order_date"] = late_rate_trend["order_date"].astype(str)
late_rate_trend["Late_delivery_risk"] *= 100  # Convert to percentage

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

# Column chart for total orders
order_counts = df.groupby(df["order_date"].dt.to_period("M"))["Order Id"].nunique().reset_index()
order_counts["order_date"] = order_counts["order_date"].astype(str)
ax2.bar(order_counts["order_date"], order_counts["Order Id"], alpha=0.3, label="Total Orders", color="gray")

# Line chart for late delivery rate
ax1.plot(late_rate_trend["order_date"], late_rate_trend["Late_delivery_risk"], color="red", marker="o", label="Late Delivery Rate (%)")

ax1.set_xlabel("Month")
ax1.set_ylabel("Late Delivery Rate (%)", color="red")
ax2.set_ylabel("Total Orders", color="gray")
ax1.set_title("Late Delivery Rate vs Total Orders Over Time")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)


# --- "Stuck" Orders by Status ---
st.subheader("🧱 'Stuck' Orders by Status")

status_counts = df["Order Status"].value_counts().reset_index()
status_counts.columns = ["Order Status", "Count"]

col1, col2 = st.columns(2)

with col1:
    st.write("### Donut Chart View")
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        status_counts["Count"],
        labels=status_counts["Order Status"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4)
    )
    ax.set_title("Distribution of Orders by Status")
    st.pyplot(fig)

with col2:
    st.write("### Bar Chart View")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(status_counts["Order Status"], status_counts["Count"], color="skyblue")
    ax.set_title("Orders by Status")
    ax.set_xlabel("Order Status")
    ax.set_ylabel("Number of Orders")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    st.pyplot(fig)

# --- Bottleneck Identification and Recommendations ---
st.subheader("🚨 Identified Bottlenecks & AI Recommendations")

bottlenecks = category_analysis[
    (category_analysis["late_delivery_rate"] > late_threshold) |
    (category_analysis["cancellation_rate"] > cancel_threshold) |
    (category_analysis["fraud_rate"] > fraud_threshold) |
    (category_analysis["Revenue_Loss"] > loss_threshold)
]

if not bottlenecks.empty:
    q_table["Best Recommendation"] = q_table.idxmax(axis=1)
    
    recommendations = q_table.loc[bottlenecks["Category Name"]][["Best Recommendation"]]
    
    # Merge recommendations with bottleneck data for context
    bottlenecks_with_rec = pd.merge(
        bottlenecks.sort_values(by="Revenue_Loss", ascending=False),
        recommendations,
        left_on="Category Name",
        right_index=True
    ).set_index("Category Name")

    st.write("The following categories are identified as bottlenecks based on your thresholds. The AI recommends the following actions to mitigate issues:")
    st.dataframe(bottlenecks_with_rec.style.format({
        'late_delivery_rate': '{:.2f}%',
        'cancellation_rate': '{:.2f}%',
        'fraud_rate': '{:.2f}%',
        'avg_lead_time': '{:.2f}',
        'avg_order_value': '${:,.2f}',
        'Revenue_Loss': '${:,.2f}'
    }).apply(lambda x: ['background-color: #FFC7CE' if x.name in ['late_delivery_rate', 'cancellation_rate', 'fraud_rate', 'Revenue_Loss'] else '' for i in x], axis=1))

else:
    st.success("✅ No bottlenecks found with the current threshold settings.")
