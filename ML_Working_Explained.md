# Predictive Maintenance — ML Working, Workflow & DFD

---

## 1. How the ML Model Works (Simplified)

### 1.1 The Goal

> Given the last 50 sensor readings from a machine, **predict whether it will fail soon**.

The model answers one question: **What is the risk level?**

| Output | Meaning |
|--------|---------|
| **LOW (0)** | Machine is healthy, no action needed |
| **MEDIUM (1)** | Showing early signs of wear, schedule maintenance |
| **HIGH (2)** | Failure is likely, immediate attention required |

### 1.2 What is a RandomForest?

Think of it as **150 experts** voting on the machine's condition:

```mermaid
graph LR
    A["Sensor Data<br/>(13 features)"] --> T1["🌳 Tree 1<br/>Votes: LOW"]
    A --> T2["🌳 Tree 2<br/>Votes: MEDIUM"]
    A --> T3["🌳 Tree 3<br/>Votes: LOW"]
    A --> T4["🌳 ..."]
    A --> T5["🌳 Tree 150<br/>Votes: LOW"]
    T1 --> V["🗳️ Majority Vote"]
    T2 --> V
    T3 --> V
    T4 --> V
    T5 --> V
    V --> R["Final Answer: LOW<br/>Confidence: 85%"]
```

**Each tree** is a Decision Tree — a flowchart of yes/no questions:

```mermaid
graph TD
    Q1{"Is vibration > 3.5g?"}
    Q1 -->|Yes| Q2{"Is temperature rising?"}
    Q1 -->|No| Q3{"Is stress_index > 45?"}
    Q2 -->|Yes| H["⚠️ HIGH RISK"]
    Q2 -->|No| M["🟡 MEDIUM RISK"]
    Q3 -->|Yes| M2["🟡 MEDIUM RISK"]
    Q3 -->|No| L["🟢 LOW RISK"]
```

**Why 150 trees?** One tree can make mistakes, but when 150 trees vote together, the errors cancel out — like asking 150 doctors instead of 1.

### 1.3 What Goes INTO the Model (Features)

Raw sensor data is **too noisy** to use directly. We engineer 13 meaningful features:

```mermaid
graph LR
    subgraph Raw["Raw Data (3 sensors)"]
        C["Current (A)"]
        T["Temperature (°C)"]
        V["Vibration (g)"]
    end
    subgraph Eng["Engineered Features (13 total)"]
        R1["3 × Latest values"]
        R2["3 × Rolling averages"]
        R3["3 × Rate of change (delta)"]
        R4["3 × Trend direction"]
        R5["1 × Stress index"]
    end
    Raw --> Eng
```

**Example**: For a machine with these last 10 vibration readings:
`[1.2, 1.3, 1.5, 1.8, 2.1, 2.5, 2.9, 3.2, 3.6, 4.0]`

| Feature | Value | What It Tells Us |
|---------|-------|------------------|
| `vibration` (latest) | 4.0g | Current state — elevated |
| `vibration_rolling_avg` | 2.61g | Average over 10 readings |
| `vibration_delta` | +0.4g | Last reading jumped by 0.4 |
| `vibration_trend` | RISING (+1) | Consistently increasing |
| [stress_index](file:///f:/Predictionmodel/analytics.py#155-209) | 62 | Overall machine strain is high |

### 1.4 How the Model Learns (Training)

Since we can't manually label thousands of readings as "healthy" or "failing", we use **Weak Supervision** — automated rules that act like a teacher:

```mermaid
graph TD
    subgraph WS["Weak Supervision Rules"]
        R1["IF vibration > 5g AND temp rising → HIGH"]
        R2["IF stress > 70 → HIGH"]
        R3["IF vibration > 3.5g → MEDIUM"]
        R4["IF temp > 63.75°C → MEDIUM"]
        R5["IF everything normal → LOW"]
    end
    D["Historical Data<br/>(500 readings/device)"] --> SW["Sliding Window<br/>(50 readings each)"]
    SW --> FE["Feature Engineering<br/>(13 features)"]
    FE --> WS
    WS --> L["Labeled Dataset<br/>(features + label)"]
    L --> RF["RandomForest<br/>Training"]
    RF --> M["Trained Model<br/>(rf_model.joblib)"]
```

**Why weak supervision?** In a factory, you can't stop machines and manually tag every reading as "about to fail". Instead, we define rules from domain knowledge (e.g., "vibration above 5g with rising temperature is dangerous") and let the algorithm learn the subtler patterns.

### 1.5 What Comes OUT (Predictions)

```mermaid
graph LR
    F["13 Features"] --> M["Trained<br/>RandomForest"]
    M --> P["Class Probabilities"]
    P --> HS["Health Score<br/>(0-100)"]
    P --> RL["Risk Level<br/>(LOW/MED/HIGH)"]
    P --> MR["Maintenance<br/>Required? (Y/N)"]
    P --> FR["Failure Reason<br/>(Human-readable)"]
```

**Health Score formula**:
```
Health = P(LOW) × 100 + P(MEDIUM) × 50 + P(HIGH) × 0
```

| Model Output | P(LOW) | P(MED) | P(HIGH) | Health | Risk | Maintenance? |
|-------------|--------|--------|---------|--------|------|-------------|
| Healthy | 0.92 | 0.06 | 0.02 | 95 | LOW | No |
| Degrading | 0.25 | 0.60 | 0.15 | 55 | MEDIUM | Yes |
| Failing | 0.03 | 0.12 | 0.85 | 9 | HIGH | Yes |

---

## 2. Complete System Workflow

### 2.1 End-to-End Workflow Diagram

```mermaid
graph TD
    subgraph HW["HARDWARE LAYER"]
        E1["ESP32 Device 1"] 
        E2["ESP32 Device 2"]
        E3["ESP32 Device N"]
    end

    subgraph DB["DATABASE LAYER"]
        FB["Firebase Realtime DB"]
        LIVE["/machines/{id}/live"]
        HIST["/machines/{id}/history"]
        PRED["/machines/{id}/predictions"]
    end

    subgraph BE["BACKEND LAYER (FastAPI)"]
        INIT["1. Startup: Init Firebase + Load Model"]
        DISC["2. Discover Devices"]
        
        subgraph LOOP["3. Scheduler Loop (Every 3-5s)"]
            PULL["Pull History (50 records)"]
            FEAT["Compute 13 Features"]
            MLPRED["ML Prediction"]
            WRITE["Write Result to Firebase"]
            CACHE["Update In-Memory Cache"]
        end
        
        subgraph API["4. API Layer"]
            REST["REST Endpoints"]
            WS["WebSocket Stream"]
            CHAT["AI Chat (Llama 3)"]
        end
    end

    subgraph FE["FRONTEND LAYER"]
        DASH["Dashboard"]
        CHARTS["Sensor Charts"]
        GAUGES["Live Gauges"]
        AI["AI Chat Panel"]
    end

    E1 & E2 & E3 -->|Write sensor data| FB
    FB --> LIVE & HIST
    INIT --> DISC
    DISC --> LOOP
    PULL -->|Read from Firebase| HIST
    PULL --> FEAT --> MLPRED --> WRITE
    WRITE -->|Write to Firebase| PRED
    MLPRED --> CACHE
    CACHE --> API
    REST & WS -->|JSON/WebSocket| DASH
    CHAT -->|LLM Response| AI
    DASH --> CHARTS & GAUGES & AI
```

### 2.2 Training Workflow

```mermaid
graph LR
    A["1. Run train_model.py"] --> B["2. Connect to Firebase"]
    B --> C["3. Pull all device history"]
    C --> D["4. Sliding window<br/>(size=50, step=5)"]
    D --> E["5. Build 13 features<br/>per window"]
    E --> F["6. Auto-label using<br/>weak supervision rules"]
    F --> G["7. Train RandomForest<br/>(150 trees, depth=12)"]
    G --> H["8. Cross-validate"]
    H --> I["9. Save model to<br/>models/rf_model.joblib"]
```

### 2.3 Inference Workflow (Real-Time)

```mermaid
graph LR
    A["Scheduler tick<br/>(every 3-5s)"] --> B["Read last 50<br/>readings from Firebase"]
    B --> C["Compute 13 features"]
    C --> D{"ML model<br/>loaded?"}
    D -->|Yes| E["RandomForest.predict()"]
    D -->|No| F["Rule-based fallback<br/>(stress index thresholds)"]
    E --> G["Build result:<br/>health, risk, reason"]
    F --> G
    G --> H["Write to Firebase"]
    G --> I["Update cache"]
    I --> J["Push to WebSocket"]
    J --> K["Dashboard updates"]
```

---

## 3. Data Flow Diagrams (DFD)

### 3.1 Context Diagram (Level 0)

Shows the entire system as a single process with external entities:

```mermaid
graph LR
    ESP["🔧 ESP32 Devices<br/>(External Entity)"]
    USER["👤 Dashboard User<br/>(External Entity)"]
    LLM["🤖 LLM Provider<br/>(Groq / Ollama)"]
    
    ESP -->|"Sensor Data<br/>(current, temp, vibration)"| SYS["🏭 Predictive<br/>Maintenance<br/>Platform"]
    SYS -->|"Device List, Charts,<br/>Status, Predictions"| USER
    USER -->|"Select device,<br/>Ask AI question"| SYS
    SYS -->|"Device context +<br/>User question"| LLM
    LLM -->|"AI diagnostic<br/>response"| SYS
```

### 3.2 Level 1 DFD — Major Processes

```mermaid
graph TD
    ESP["🔧 ESP32"] -->|Raw sensor data| DS1[("Firebase<br/>RTDB")]
    
    DS1 -->|Device IDs| P1["P1: Device<br/>Discovery"]
    P1 -->|Registered devices| DS2[("Device<br/>Registry")]
    
    DS1 -->|History (50 records)| P2["P2: Feature<br/>Engineering"]
    P2 -->|13 features| P3["P3: ML<br/>Prediction"]
    
    P3 -->|Prediction result| DS1
    P3 -->|Cached prediction| DS3[("Prediction<br/>Cache")]
    
    DS3 -->|Latest prediction| P4["P4: API<br/>Server"]
    DS2 -->|Device info| P4
    DS1 -->|Chart data| P4
    
    P4 -->|"JSON / WebSocket"| USER["👤 User"]
    
    USER -->|"Chat message"| P5["P5: Context<br/>Builder"]
    DS2 -->|Device data| P5
    DS3 -->|Prediction| P5
    P5 -->|Grounded context| P6["P6: Chat<br/>Engine"]
    P6 -->|Prompt| LLM["🤖 LLM"]
    LLM -->|Response| P6
    P6 -->|AI answer| P4
```

### 3.3 Level 2 DFD — ML Prediction Process (P3 expanded)

```mermaid
graph TD
    IN["13 Features<br/>(from P2)"] --> CHECK{"Model<br/>loaded?"}
    
    CHECK -->|Yes| ML_PATH
    CHECK -->|No| RULE_PATH
    
    subgraph ML_PATH["ML Prediction Path"]
        A1["Convert features<br/>to numpy array"]
        A2["RandomForest<br/>.predict()"]
        A3["Get class<br/>probabilities"]
        A4["Compute health<br/>score from probs"]
        A1 --> A2 --> A3 --> A4
    end
    
    subgraph RULE_PATH["Rule-Based Fallback Path"]
        B1["Read stress_index"]
        B2{"stress > 70?"}
        B3{"stress > 45?"}
        B4["HIGH risk"]
        B5["MEDIUM risk"]
        B6["LOW risk"]
        B1 --> B2
        B2 -->|Yes| B4
        B2 -->|No| B3
        B3 -->|Yes| B5
        B3 -->|No| B6
    end
    
    A4 --> OUT
    B4 & B5 & B6 --> OUT
    
    OUT["Prediction Result"] --> WR["Write to Firebase"]
    OUT --> CA["Update Cache"]
    OUT --> RE["Update Registry"]
```

### 3.4 Level 2 DFD — Feature Engineering Process (P2 expanded)

```mermaid
graph TD
    IN["50 sensor readings<br/>(from Firebase)"] --> S1["Extract latest<br/>values (3)"]
    IN --> S2["Compute rolling<br/>averages (3)"]
    IN --> S3["Compute deltas<br/>(3)"]
    IN --> S4["Detect trends via<br/>linear regression (3)"]
    IN --> S5["Calculate composite<br/>stress index (1)"]
    
    S1 & S2 & S3 & S4 & S5 --> MERGE["Merge into<br/>feature vector"]
    MERGE --> OUT["13 Features<br/>(Dict → Array)"]
```

---

## 4. ML Pipeline Summary Table

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **Data Collection** | Firebase RTDB | Pull 500 records per device | Raw time-series data |
| **Windowing** | Raw data | Sliding window (size=50, step=5) | Multiple overlapping samples |
| **Feature Engineering** | 50 raw readings | Rolling avg, delta, trend, stress | 13 numeric features |
| **Labeling** | 13 features | Threshold rules (weak supervision) | Label: 0, 1, or 2 |
| **Training** | Features + Labels | RandomForest (150 trees, depth=12) | Trained classifier |
| **Validation** | Training data | k-fold cross-validation | Accuracy score |
| **Serialization** | Trained model | `joblib.dump()` | `rf_model.joblib` file |
| **Loading** | `.joblib` file | `joblib.load()` at startup | In-memory model |
| **Inference** | 13 live features | `model.predict()` + `predict_proba()` | Risk + Health Score |
| **Fallback** | Stress index | Threshold comparison | Risk + Health Score |

---

## 5. Key ML Concepts Used

| Concept | How We Use It |
|---------|--------------|
| **Supervised Learning** | Model learns from labeled examples (features → risk class) |
| **Ensemble Methods** | 150 trees vote together → more accurate than 1 tree |
| **Bagging** | Each tree trained on a random subset of data → reduces overfitting |
| **Feature Importance** | RandomForest tells us which sensors matter most |
| **Class Imbalance Handling** | `class_weight="balanced"` gives rare classes (HIGH risk) more weight |
| **Cross-Validation** | Test model on unseen splits to estimate real-world accuracy |
| **Weak Supervision** | Auto-generate labels from domain rules instead of manual tagging |
| **Graceful Degradation** | If ML model missing → fall back to rule-based heuristics |

---

