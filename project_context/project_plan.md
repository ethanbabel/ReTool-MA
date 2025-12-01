# **Project Proposal: Multi-Agent ReTool — Role-Decomposed Reinforcement Learning for Reliable Code-Aware Tool Use in LLMs**

## **1. Overview**

Large Language Models (LLMs) increasingly rely on **tool use** (e.g., code execution, calculators, search APIs) to solve complex tasks.  
The recent **ReTool** framework (ByteDance Seed, 2025) demonstrates that reinforcement learning over tool-augmented trajectories can significantly improve reasoning performance on challenging benchmarks.

However, ReTool exhibits a key failure mode: **Incorrect reasoning trajectories show a rapidly collapsing code pass rate**, diverging sharply from the code pass rate on correct trajectories.

This suggests that combining **reasoning**, **tool planning**, **code generation**, and **verification** into a *single LLM policy* potentially leads to cognitive overload, compounding errors, and training instability.

This project proposes **Multi-Agent ReTool (ReTool-MA)** — a novel, role-decomposed RL system that separates reasoning, code execution, and verification across specialized LLM agents. This decomposition is designed to improve code reliability and overall reasoning accuracy.

The project will empirically evaluate this multi-agent improvement using **MARTI**, a multi-agent reinforcement learning framework for tool-using LLM systems.

---

## **2. Motivation**

### **2.1 Limitations of Single-Agent ReTool**

ReTool uses a single LLM to:
1. Produce high-level reasoning  
2. Decide which tools to call  
3. Write executable code  
4. Interpret execution results  
5. Update its reasoning  

This entanglement causes:
- High cognitive load  
- Poor credit assignment across disparate skills  
- Unstable gradients for RL  
- Divergence in code-use reliability on incorrect trajectories  
- Error amplification in long-horizon reasoning (e.g., AIME math problems)

### **2.2 Why Multi-Agent**

Multi-agent systems allow **division of cognitive labor**:
- Reduce entropy of each policy  
- Stabilize training  
- Improve code correctness  
- Reduce invalid tool calls  
- Provide natural redundancy for verification

This aligns with principles in hierarchical RL, options frameworks, and cooperative multi-agent RL.

---

## **3. Proposed Method: Multi-Agent ReTool (ReTool-MA)**

ReTool-MA decomposes ReTool's monolithic policy into three specialized LLM agents:

### **3.1 Planner Agent**

**Role:**  
- Generate reasoning steps  
- Produce structured subgoals  
- Decide when to call code execution  
- Provide specifications (e.g., “compute gcd(210, 840)”)  

**Inputs:** prompt, problem statement  
**Outputs:** reasoning; tool call specifications  
**Training:** RL via MARTI (policy gradient, PPO-variant)

### **3.2 Executor Agent**

**Role:**  
- Convert Planner specifications into *executable code*  
- Ensure syntactic validity  
- Output code and resulting Python execution trace  
- Never engages in reasoning  

**Inputs:** tool plan or function-call spec  
**Outputs:** Python code + execution results  
**Training:**  
- Lightweight SFT for code formatting  
- Optionally RL auxiliary objective: maximize pass rate

### **3.3 Verifier Agent**

**Role:**  
- Cross-check the Planner’s reasoning  
- Validate Executor outputs  
- Flag incorrect logic, inconsistent values  
- Approve or ask for new reasoning / new code  

**Training:** RL reward for correct final answers + correct verification

---

## **4. RL Training Setup**

### **4.1 Environment: MARTI**

MARTI provides:
- Multi-agent rollout orchestration  
- Code execution tools  
- Task environments: GSM8K, AIME, BigBench Hard, math proofs, etc.  
- Trajectory-level rewards  
- Policy optimization frameworks (GRPO, PPO, tempered policy gradients)

### **4.2 Action Space**

| Agent | Action Type | Description |
|-------|-------------|-------------|
| Planner | Free-form reasoning + function-call specs | Natural language + structured tool actions |
| Executor | Code generation | Returns python code directly |
| Verifier | Binary + natural language | Accept/Reject reasoning, request tool re-run |

### **4.3 Rewards**

Same as seen in ReTool, with additional rewards tuning for Multi-Agent framework TBD.

---

## **5. Hypotheses**

1. **H1:** ReTool-MA will exhibit **higher code pass rates**, especially on incorrect mid-trajectory reasoning attempts (fixing ReTool’s failure mode).

2. **H2:** ReTool-MA will outperform single-agent ReTool on **AIME** tasks.

3. **H3:** Multi-agent decomposition will reduce:
   - Invalid code calls  
   - Execution errors  
   - Degenerate long-horizon reasoning loops  

4. **H4:** Role specialization improves RL stability and sample efficiency.

---

## **6. Experiment Design**

### **Datasets:**
- **AIME (primary)**  
- Optional: GSM8K  
- Optional: Math reasoning tasks with Python-tools  
- Optional: MATH and BigBench-Hard  

### **Baselines:**
1. Single-agent ReTool (reproduction using MARTI)  
2. Naive SFT tool-using model  

### **Metrics:**
- **Final Accuracy** (AIME score)  
- **Code Execution Pass Rate**  
- **Number of tool calls per problem**  
- **Invalid code call rate**  
- **Episode length / number of reasoning steps**  
- **Reward convergence** (training stability)