# AI Workflow Document (AI_Workflow.md)
# Indian Climate Digital Twin (ICDT) - Deep Research Platform
## Aligned with the Bharatiya Antariksh Hackathon (BAH) 2026 — Challenge 05

**Version:** 3.0
**Date:** 2026-06-28
**Status:** Pitch Proposal / Draft
**Team Context:** Student Innovators (Undergraduate/Postgraduate/PhD Research Scholars)
**Hackathon Event:** Bharatiya Antariksh Hackathon (BAH) 2026 (ISRO & Hack2skill)
**Challenge Statement:** Challenge 05 — AI-Powered Digital Twin of India's Climate
**Submission Stage:** Concept Proposal / Idea Submission (Deadline: July 1, 2026)
**Grand Finale Timeline:** August 6–7, 2026 (30-hour offline hackathon with ISRO scientist mentorship)

---

## 1. AI Workflow Philosophy

The ICDT AI Workflow is designed around the principle of **"Human-AI Symbiosis for Climate Intelligence."** Rather than replacing human expertise, the AI system amplifies the capabilities of climate scientists, policymakers, and farmers by automating data-intensive tasks while preserving human judgment for critical decisions. 

In the context of the **Bharatiya Antariksh Hackathon 2026 (BAH2026)**, the workflow is engineered to ingest sovereign ISRO and IMD observations and run fast, resource-optimized simulations of Indian climate variability, agricultural forecasting, and urban resilience.

### 1.1 Challenge Objectives & Focus Areas
Under the BAH2026 guidelines, our solution focuses on four key climate-tech pillars:
1. **Multi-Source Data Integration:** Fusing ISRO satellite datasets (INSAT-3D/3DR Imager and Sounder profiles, Oceansat-3, SCATSAT-1, Cartosat/Resourcesat/RISAT-1A), NICES Essential Climate Variables (ECVs), Bhuvan geospatial/hazard layers, and IMD historical grids (124-year rainfall, 73-year temperature).
2. **"What-If" Counterfactual Simulation:** Interactive prediction of climate changes under various temperature fluctuations, rainfall variability, greenhouse gas (CO2) concentrations, and Land Use/Land Cover (LULC) changes.
3. **Sectoral Impact Mapping:** Actionable intelligence for agriculture (irrigation, crop yield, pest risk), hydrology (river basin streamflow, reservoir levels), and urban resilience (flood hazards, heatwaves).
4. **Explainability & Performance:** High-throughput inference suitable for edge deployment at Krishi Vigyan Kendras (KVKs) and municipal command centers, with complete transparency via SHAP, Grad-CAM, and physical consistency audits.

### 1.2 Design Principles
1. **Explainability First:** Every AI prediction must be accompanied by a human-readable explanation, confidence score, and uncertainty quantification
2. **Physics-Grounded:** ML models must respect physical laws (conservation of mass, energy, momentum); violations trigger automatic model retraining
3. **Continuous Learning:** The system learns from new observations, user feedback, and research publications in real-time
4. **Multi-Agent Collaboration:** No single model makes decisions; consensus across physics, data-driven, and expert models is required
5. **Ethical AI:** Fairness across regions, income levels, and languages; no harm to vulnerable populations
6. **Open Science:** Research artifacts, model weights, and training data are open by default (with privacy safeguards)
7. **Sovereign & Cost-Aware Design:** 100% reliance on national data services (Bhuvan, MOSDAC, IMDLIB) and low-carbon, GPU-optimized pipelines (using `codecarbon` instrumentation) to meet rigorous government and technical evaluations.

---

## 2. Deep Research Agent (DRA) Workflow

### 2.1 Overview

The Deep Research Agent is the cognitive engine of ICDT. It autonomously conducts climate research by planning, executing, and synthesizing multi-source investigations. The workflow is implemented as a **stateful multi-agent system** using LangGraph with persistent checkpoints.

### 2.2 Workflow Stages

```
STAGE 0: TRIAGE and CLARIFICATION
  Input: User query (e.g., "Impact of delayed monsoon on Punjab rice")
  Triage Agent: Classify query type (research / policy / advisory)
  Clarifier Agent: Ask follow-up questions if context is missing
  Output: Enriched query with constraints, audience, format, depth

STAGE 1: PLANNING (DAG Generation)
  Planner Agent:
    1. Exploratory search to map topic landscape
    2. Identify research dimensions (physical, social, economic)
    3. Generate sub-questions with dependency graph
    4. Assign source dimensions (academic, policy, media, data)
  Output: Directed Acyclic Graph (DAG) of research questions
  HITL: Pause for user confirmation/modification of DAG

STAGE 2: PARALLEL RESEARCH (Map/Reduce)
  Master Agent:
    1. Topological sort of DAG -> identify independent nodes
    2. Dispatch Researcher Agents in parallel (up to 10 concurrent)
    3. Each Researcher Agent:
       a. Deconstruct question into search queries
       b. Execute multi-source search (web + internal + data)
       c. Rerank and diversify results (cross-encoder + MMR)
       d. Extract, chunk, and synthesize content
       e. Identify gaps and confidence score
    4. Collect all QuestionAnalysis in shared state
  Output: Array of QuestionAnalysis with findings, gaps, citations

STAGE 3: COVERAGE ASSESSMENT and ITERATION
  Refiner Agent:
    1. Assess coverage across all research dimensions (target: 85%)
    2. Identify under-covered dimensions
    3. Generate targeted follow-up queries
    4. If coverage < 85% AND iterations < 5: loop to Stage 2
    5. If coverage >= 85% OR max iterations: proceed to Stage 4
  Output: Final consolidated research findings with gap analysis

STAGE 4: SYNTHESIS and WRITING
  Writer Agent:
    1. Meta-analysis: identify emergent themes across all findings
    2. Generate dynamic outline based on themes (not pre-defined template)
    3. Parallel section writing (one section per theme)
    4. Citation management and consistency checks
    5. Self-review for hallucinations and unsupported claims
    6. Uncertainty quantification for each major finding
  Output: Structured research artifact (report, memo, or dashboard)

STAGE 5: HUMAN REVIEW and PUBLICATION
  HITL Review:
    1. Domain expert reviews findings and citations
    2. Fact-checker verifies all claims against primary sources
    3. Policy analyst reviews recommendations for feasibility
    4. Ethics board reviews for fairness and bias
  Publication:
    - Internal: Knowledge graph update, model card update
    - External: Open data portal, research preprint, policy brief
```

### 2.3 State Management

The entire workflow is managed through a **SharedLangGraphState** with the following structure:

```python
class DeepResearchState:
    # Input
    user_query: str
    enriched_query: Optional[EnrichedQuery]

    # Planning
    research_dag: Optional[DAG]
    dag_approved: bool = False

    # Research
    question_analyses: List[QuestionAnalysis]
    current_iteration: int = 0
    max_iterations: int = 5

    # Coverage
    coverage_assessment: Optional[CoverageReport]
    follow_up_queries: List[str]

    # Synthesis
    emergent_themes: List[Theme]
    report_outline: Optional[Outline]
    sections: List[Section]

    # Output
    final_artifact: Optional[ResearchArtifact]
    review_status: str = "pending"  # pending, approved, rejected, revised

    # Metadata
    thread_id: str
    checkpoint_id: Optional[str]
    cost_tracker: CostTracker
    timeline: List[TimelineEvent]
```

### 2.4 Agent Prompts

#### 2.4.1 Triage Agent Prompt

Role: Triage Agent for ICDT Deep Research system. Classify incoming queries and determine if clarification is needed.

Query Types:
- RESEARCH: Academic/scientific inquiry requiring literature synthesis
- POLICY: Government decision-support requiring data-backed recommendations
- ADVISORY: Operational guidance for farmers, disaster managers, or city planners
- EXPLORATORY: Open-ended exploration of climate topics

Clarification Triggers:
- Missing temporal scope (e.g., "when?" - monsoon season, decadal, historical)
- Missing spatial scope (e.g., "where?" - district, state, river basin, national)
- Missing audience (e.g., "for whom?" - Cabinet, farmers, researchers, public)
- Missing output format (e.g., "how?" - memo, paper, dashboard, presentation)
- Ambiguous terminology (e.g., "monsoon" could mean SW monsoon, NE monsoon, or specific months)

Constraints: Ask at most 3 clarifying questions. Prioritize questions that most affect research quality. Suggest default values if user does not specify.

Output Format: JSON with query_type, needs_clarification, clarifying_questions, enriched_query

#### 2.4.2 Planner Agent Prompt

Role: Planner Agent for ICDT Deep Research. Create structured research plans.

Task:
1. Perform a brief exploratory search to understand the topic landscape
2. Identify 4-6 key research dimensions relevant to Indian climate science
3. Generate a DAG of sub-questions with explicit dependencies
4. Assign each node a priority and source dimension

Research Dimensions (typical for climate topics):
- Physical Drivers: Atmospheric, oceanic, land surface processes
- Historical Context: Past events, trends, variability
- Impacts: Agriculture, water, health, infrastructure, economy
- Adaptation: Strategies, policies, technologies, community resilience
- Future Projections: Climate models, scenarios, uncertainty
- Governance: Institutional frameworks, international agreements, funding

DAG Rules:
- Each node must have a unique ID
- Dependencies must form a DAG (no cycles)
- Nodes with no dependencies can run in parallel
- Max 15 nodes for standard depth, 30 for comprehensive

Source Dimensions: ACADEMIC, POLICY, MEDIA, DATA

#### 2.4.3 Researcher Agent Prompt

Role: Researcher Agent for ICDT Deep Research. Investigate a specific sub-question.

Task:
1. Read assigned question and any prior findings from shared state
2. Formulate 3-5 strategic search queries using:
   - Question deconstruction (break into logical components)
   - Keyword deepening (extract entities from prior findings)
   - Gap-driven inquiry (target identified gaps from previous research)
3. Execute searches across multiple sources
4. Rerank results by relevance using cross-encoder
5. Extract and semantically chunk content
6. Synthesize findings with proper citations
7. Identify gaps for downstream researchers

Search Sources:
- Google Search API (general web)
- Semantic Scholar / arXiv (academic)
- ICDT Internal Knowledge Graph (Neo4j)
- ICDT Data Catalog (CKAN)
- Exa API (deep semantic search)

Quality Standards:
- Every claim must have a citation
- Prefer primary sources over secondary summaries
- Note confidence level for each finding
- Flag contradictory evidence explicitly
- Identify methodological limitations

Stopping Criteria:
- Coverage threshold: 85% of assigned dimension
- Max iterations: 5 search-refine loops
- Confidence threshold: 0.80

#### 2.4.4 Writer Agent Prompt

Role: Writer Agent for ICDT Deep Research. Synthesize research findings into polished artifacts.

Task:
1. Perform meta-analysis across all QuestionAnalysis to identify emergent themes
2. Generate a dynamic outline based on themes (not a pre-defined template)
3. Write each section in parallel using divide-and-conquer
4. Ensure citation consistency and completeness
5. Add executive summary, recommendations, and appendices
6. Self-review for hallucinations and unsupported claims
7. Include uncertainty quantification for each major finding

Quality Gates:
- Every claim must have >=1 citation
- Confidence score for each major finding
- Explicit uncertainty quantification
- Contradictory evidence presented fairly
- Methodological limitations acknowledged
- Recommendations are actionable and specific to India

Audience Adaptation:
- CABINET: Executive summary first, bullet-point recommendations, cost estimates
- RESEARCHER: Full methodology, data tables, statistical significance, future work
- FARMER: Simple language, visual diagrams, actionable steps, regional language
- PUBLIC: Infographics, analogies, no jargon, positive framing

---

## 3. Climate Simulation AI Workflow

### 3.1 Physics-Informed Neural Network (PINN) Training Workflow

```
INPUT: Historical climate data (ERA5, IMD, ISRO) + Physics equations

STAGE 1: DATA PREPARATION
  - Extract atmospheric fields (u, v, T, q, p) from reanalysis
  - Generate training pairs: (initial_state, evolved_state)
  - Create coordinate grids (lat, lon, pressure_level, time)
  - Normalize features (z-score per variable per level)
  - Split: 80% train, 10% validation, 10% test (temporal holdout)

STAGE 2: MODEL ARCHITECTURE
  Fourier Neural Operator (FNO):
    - Modes: 12 (spherical harmonics for global structure)
    - Width: 32 (channel dimension)
    - Depth: 4 (Fourier layers)
    - Activation: GELU
  Physics Loss:
    - Navier-Stokes residual (momentum equation)
    - Continuity equation (mass conservation)
    - Thermodynamic equation (energy conservation)
    - Moisture equation (water conservation)

STAGE 3: TRAINING LOOP
  For each epoch:
    1. Forward pass: predict evolved state from initial state
    2. Data loss: MSE(prediction, target)
    3. Physics loss: PDE residual at prediction
    4. Boundary loss: Enforce boundary conditions
    5. Total loss: L_data + 0.1*L_physics + 0.01*L_boundary
    6. Backpropagation and optimizer step (AdamW, lr=1e-4)
    7. Validation: Check generalization on held-out period
    8. Early stopping: patience=10 epochs

STAGE 4: PHYSICAL PLAUSIBILITY CHECKS
  Automated checks (fail = model rejection):
    - Mass conservation: Global integral of divergence < 1e-6
    - Energy conservation: Total energy change < 0.1% per day
    - Monotonicity: Temperature decreases with height (lapse rate)
    - Bounds: No negative humidity, wind speed < 200 m/s
    - CFL condition: Stable for chosen timestep

STAGE 5: DEPLOYMENT and MONITORING
  - Export to ONNX for inference optimization
  - Deploy on KServe with auto-scaling
  - Monitor: Data drift, concept drift, physical plausibility
  - Retrigger training if drift > threshold OR physical check fails
```

### 3.2 Ensemble Forecasting Workflow

```python
class EnsembleForecastingWorkflow:
    def __init__(self):
        self.nwp_model = WRFModel()  # Traditional physics model
        self.pinn_surrogate = WeatherPINN()  # Neural surrogate
        self.data_assimilation = EnsembleKalmanFilter()

    def generate_ensemble(self, initial_conditions, ensemble_size=50):
        # Step 1: Perturb initial conditions
        perturbed_ics = []
        for i in range(ensemble_size):
            perturbation = self.generate_perturbation(
                covariance=self.observation_error_covariance,
                method="bred_vector"  # Bred Vector Growth
            )
            perturbed_ics.append(initial_conditions + perturbation)

        # Step 2: Run ensemble members (parallel on HPC)
        ensemble_members = []
        for ic in perturbed_ics:
            if self.use_pinn:
                forecast = self.pinn_surrogate.predict(ic, lead_time=72)
            else:
                forecast = self.nwp_model.run(ic, lead_time=72)
            ensemble_members.append(forecast)

        # Step 3: Data assimilation at intermediate times
        for obs_time in self.observation_times:
            observations = self.fetch_observations(obs_time)
            ensemble_members = self.data_assimilation.update(
                ensemble=ensemble_members,
                observations=observations,
                observation_operator=self.H
            )

        # Step 4: Post-processing and uncertainty quantification
        ensemble_mean = np.mean(ensemble_members, axis=0)
        ensemble_spread = np.std(ensemble_members, axis=0)

        # Step 5: Probabilistic products
        probabilities = {
            "heavy_rain": np.mean(
                [m.precipitation > 100 for m in ensemble_members], axis=0
            ),
            "heatwave": np.mean(
                [m.temperature_max > 45 for m in ensemble_members], axis=0
            ),
            "cyclone_formation": self.detect_cyclone(ensemble_members)
        }

        return EnsembleForecast(
            mean=ensemble_mean,
            spread=ensemble_spread,
            members=ensemble_members,
            probabilities=probabilities,
            valid_times=self.valid_times
        )
```

---

## 4. Agricultural Advisory AI Workflow

### 4.1 Farm Digital Twin Update Loop

```python
class FarmAdvisoryWorkflow:
    def __init__(self, farm_id: str):
        self.farm = FarmDigitalTwin(farm_id)
        self.models = {
            "soil_moisture": SoilMoistureModel(),
            "crop_growth": DSSATModel(),
            "pest_risk": PestPredictionModel(),
            "yield_forecast": YieldPredictionModel()
        }
        self.recommender = AdvisoryRecommender()

    def daily_update(self):
        # Step 1: Fetch latest observations
        observations = self.fetch_observations()
        # Sources: Sentinel-1 (soil moisture), Sentinel-2 (NDVI),
        #          IMD AWS (weather), farmer app (crop stage, irrigation log)

        # Step 2: Update farm twin state
        self.farm.update_state(observations)

        # Step 3: Run predictive models
        soil_moisture_forecast = self.models["soil_moisture"].predict(
            current=self.farm.soil_moisture,
            weather_forecast=self.farm.weather_forecast_7day,
            horizon=7
        )

        crop_stage_forecast = self.models["crop_growth"].simulate(
            current_stage=self.farm.crop_stage,
            weather=self.farm.weather_forecast_7day,
            water=soil_moisture_forecast
        )

        pest_risk = self.models["pest_risk"].predict(
            weather=self.farm.weather_forecast_7day,
            crop_stage=crop_stage_forecast,
            historical_outbreaks=self.farm.pest_history
        )

        yield_forecast = self.models["yield_forecast"].predict(
            crop_stage=crop_stage_forecast,
            weather=self.farm.weather_forecast_30day,
            soil=self.farm.soil_profile,
            management=self.farm.management_practices
        )

        # Step 4: Generate recommendations
        advisory = self.recommender.generate(
            farm=self.farm,
            soil_moisture=soil_moisture_forecast,
            crop_stage=crop_stage_forecast,
            pest_risk=pest_risk,
            yield_forecast=yield_forecast,
            constraints={
                "water_budget": self.farm.water_rights,
                "labor_availability": self.farm.labor_calendar,
                "market_prices": self.farm.commodity_prices
            }
        )

        # Step 5: Personalize and translate
        personalized = self.personalize_advisory(
            advisory=advisory,
            farmer_profile=self.farm.farmer_profile,
            language=self.farm.farmer_profile.preferred_language,
            literacy_level=self.farm.farmer_profile.literacy_level
        )

        # Step 6: Deliver via appropriate channel
        self.deliver_advisory(
            advisory=personalized,
            channels=self.farm.farmer_profile.preferred_channels
            # SMS, WhatsApp, Voice call, Mobile app push, Krishi Vigyan Kendra
        )

        # Step 7: Log and learn
        self.log_advisory(advisory)
        self.collect_feedback()  # Farmer feedback on advisory usefulness

        return personalized
```

### 4.2 Voice-Based Advisory Workflow

```python
class VoiceAdvisoryWorkflow:
    def __init__(self):
        self.asr = IndicWav2VecModel()  # Automatic Speech Recognition
        self.nlp = IndicBERTModel()      # Natural Language Understanding
        self.tts = AI4BharatTTS()        # Text-to-Speech

    def handle_incoming_call(self, farmer_phone: str, language: str):
        # Step 1: Greeting and authentication
        greeting = self.tts.synthesize(
            text="Namaste! Welcome to Krishi Mitra. Please say your farmer ID.",
            language=language
        )
        play_audio(greeting)

        # Step 2: Authenticate farmer
        farmer_id_audio = record_audio(max_duration=10)
        farmer_id = self.asr.transcribe(farmer_id_audio, language=language)
        farmer = self.authenticate(farmer_id)

        if not farmer:
            error_msg = self.tts.synthesize(
                text="Sorry, I could not find your record. Please contact your Krishi Vigyan Kendra.",
                language=language
            )
            play_audio(error_msg)
            return

        # Step 3: Understand farmer's query
        query_prompt = self.tts.synthesize(
            text="How can I help you today? You can ask about irrigation, pests, weather, or market prices.",
            language=language
        )
        play_audio(query_prompt)

        query_audio = record_audio(max_duration=30)
        query_text = self.asr.transcribe(query_audio, language=language)

        # Step 4: Intent classification
        intent = self.nlp.classify_intent(query_text)
        # Intents: IRRIGATION, PEST, WEATHER, MARKET, GENERAL

        # Step 5: Retrieve relevant advisory
        advisory = self.retrieve_advisory(
            farmer=farmer,
            intent=intent,
            query=query_text
        )

        # Step 6: Synthesize and play response
        response = self.tts.synthesize(
            text=advisory.text,
            language=language,
            speed="slow",  # For better comprehension
            emphasis="key_points"
        )
        play_audio(response)

        # Step 7: Offer follow-up
        follow_up = self.tts.synthesize(
            text="Would you like more details? Say yes or no.",
            language=language
        )
        play_audio(follow_up)

        # Step 8: Log interaction
        self.log_interaction(
            farmer=farmer,
            query=query_text,
            intent=intent,
            advisory=advisory,
            duration=call_duration
        )
```

---

## 5. Urban Resilience AI Workflow

### 5.1 Real-Time Flood Prediction Workflow

```python
class FloodPredictionWorkflow:
    def __init__(self, city: str):
        self.city = city
        self.rainfall_nowcaster = RainfallNowcaster()
        self.hydraulic_model = UrbanFloodModel()
        self.impact_assessor = FloodImpactAssessor()
        self.alert_system = MultiChannelAlertSystem()

    def monitor_and_predict(self):
        while True:
            # Step 1: Ingest real-time rainfall observations
            radar_rainfall = self.fetch_radar_data(city)
            gauge_rainfall = self.fetch_gauge_data(city)
            satellite_rainfall = self.fetch_satellite_estimates(city)

            # Step 2: Merge rainfall estimates (Kalman filter)
            merged_rainfall = self.merge_rainfall(
                radar=radar_rainfall,
                gauge=gauge_rainfall,
                satellite=satellite_rainfall
            )

            # Step 3: Nowcast rainfall (0-6 hours)
            rainfall_forecast = self.rainfall_nowcaster.predict(
                current=merged_rainfall,
                lead_time=6,
                ensemble=True
            )

            # Step 4: Check if flood threshold exceeded
            if rainfall_forecast.max_accumulation > self.city.flood_threshold:

                # Step 5: Run hydraulic model
                flood_map = self.hydraulic_model.simulate(
                    rainfall=rainfall_forecast,
                    dem=self.city.dem,
                    drainage=self.city.drainage_network,
                    land_use=self.city.land_use
                )

                # Step 6: Assess impact
                impact = self.impact_assessor.assess(
                    flood_map=flood_map,
                    buildings=self.city.buildings,
                    population=self.city.population_grid,
                    critical_infrastructure=self.city.critical_infrastructure
                )

                # Step 7: Generate alert levels
                alert_level = self.determine_alert_level(impact)
                # Levels: GREEN, YELLOW, ORANGE, RED

                # Step 8: Issue targeted alerts
                self.alert_system.issue(
                    level=alert_level,
                    affected_areas=impact.affected_wards,
                    affected_population=impact.affected_population,
                    recommended_actions=self.get_recommended_actions(alert_level),
                    channels=["SMS", "WhatsApp", "Siren", "TV", "Radio", "App"]
                )

                # Step 9: Notify emergency responders
                if alert_level in ["ORANGE", "RED"]:
                    self.notify_emergency_services(
                        city=self.city,
                        impact=impact,
                        resources_needed=self.estimate_resources(impact)
                    )

                # Step 10: Update dashboard
                self.update_dashboard(
                    city=self.city,
                    rainfall=rainfall_forecast,
                    flood_map=flood_map,
                    impact=impact,
                    alert_level=alert_level
                )

            # Sleep until next observation cycle (5 minutes)
            time.sleep(300)
```

### 5.2 Heatwave Early Warning Workflow

```python
class HeatwaveWarningWorkflow:
    def __init__(self):
        self.temperature_forecaster = TemperatureEnsembleModel()
        self.vulnerability_model = HeatVulnerabilityModel()
        self.health_impact_model = HealthImpactModel()

    def generate_heatwave_warning(self, city: str, lead_time: int = 72):
        # Step 1: Temperature forecast
        temp_forecast = self.temperature_forecaster.predict(
            city=city,
            lead_time=lead_time,
            ensemble_size=50
        )

        # Step 2: Identify heatwave days
        heatwave_days = self.identify_heatwave_days(
            max_temp=temp_forecast.max_temperature,
            threshold=city.heatwave_threshold,  # Typically 45C for plains, 40C for hills
            min_duration=2  # At least 2 consecutive days
        )

        if not heatwave_days:
            return None  # No heatwave predicted

        # Step 3: Vulnerability assessment (ward-level)
        vulnerability = self.vulnerability_model.assess(
            city=city,
            demographics=city.demographics,  # elderly population, slum density
            infrastructure=city.infrastructure,  # AC penetration, green cover
            health_facilities=city.health_facilities,
            historical_mortality=city.heat_mortality_history
        )

        # Step 4: Health impact prediction
        health_impact = self.health_impact_model.predict(
            temperature=temp_forecast,
            humidity=temp_forecast.humidity,
            vulnerability=vulnerability,
            historical_health_data=city.health_records
        )

        # Step 5: Generate targeted warnings
        warnings = []
        for ward in city.wards:
            ward_vulnerability = vulnerability[ward.id]
            ward_impact = health_impact[ward.id]

            warning = HeatwaveWarning(
                ward=ward.name,
                alert_level=self.map_to_alert_level(ward_vulnerability, ward_impact),
                start_date=heatwave_days[0],
                end_date=heatwave_days[-1],
                max_temperature=temp_forecast[ward].max,
                vulnerable_population=ward.vulnerable_population,
                recommended_actions=self.get_actions(ward_vulnerability),
                cooling_centers=self.find_cooling_centers(ward),
                health_advisories=self.get_health_advisories(ward_impact)
            )
            warnings.append(warning)

        # Step 6: Issue warnings
        for warning in warnings:
            if warning.alert_level in ["ORANGE", "RED"]:
                self.issue_immediate_alert(warning)
            else:
                self.schedule_advisory(warning)

        return warnings
```

---

## 6. Ganges Basin Digital Twin Workflow

### 6.1 Hydrological Simulation Workflow

```python
class GangesBasinWorkflow:
    def __init__(self):
        self.area_km2 = 860000  # 860,000 km2 basin
        self.hydrology_model = SWATPlusModel()
        self.groundwater_model = MODFLOWModel()
        self.data_assimilation = DataAssimilationFramework()
        self.ai_enhancer = BasinAIEnhancer()

    def daily_simulation(self):
        # Step 1: Ingest observations
        precipitation = self.fetch_precipitation()  # IMD + GPM + ground gauges
        temperature = self.fetch_temperature()  # IMD gridded
        streamflow = self.fetch_streamflow()  # CWC stations
        groundwater = self.fetch_groundwater()  # CGWB monitoring wells
        irrigation = self.fetch_irrigation()  # Remote sensing (OpenET) + census

        # Step 2: Run hydrological model
        water_balance = self.hydrology_model.simulate(
            precipitation=precipitation,
            temperature=temperature,
            irrigation=irrigation,
            land_use=self.land_use_map,
            soil=self.soil_map,
            elevation=self.dem
        )

        # Step 3: Run groundwater model
        groundwater_state = self.groundwater_model.simulate(
            recharge=water_balance.groundwater_recharge,
            pumping=irrigation.groundwater_pumping,
            boundaries=self.aquifer_boundaries
        )

        # Step 4: Data assimilation (update with observations)
        updated_state = self.data_assimilation.update(
            model_state={
                "surface_water": water_balance,
                "groundwater": groundwater_state
            },
            observations={
                "streamflow": streamflow,
                "groundwater_level": groundwater
            }
        )

        # Step 5: AI-enhanced gap filling and prediction
        enhanced_state = self.ai_enhancer.enhance(
            state=updated_state,
            remote_sensing=self.fetch_remote_sensing(),  # GRACE, SMAP, MODIS
            historical_patterns=self.historical_climatology
        )

        # Step 6: Generate basin health indicators
        indicators = {
            "water_stress_index": self.calculate_water_stress(enhanced_state),
            "groundwater_depletion_rate": self.calculate_depletion(enhanced_state),
            "ecological_flow_status": self.check_ecological_flows(enhanced_state),
            "agricultural_water_productivity": self.calculate_productivity(enhanced_state),
            "flood_risk": self.assess_flood_risk(enhanced_state)
        }

        # Step 7: Update Water Cycle Atlas
        self.update_atlas(
            date=datetime.now(),
            state=enhanced_state,
            indicators=indicators
        )

        # Step 8: Generate alerts if thresholds exceeded
        if indicators["water_stress_index"] > 0.7:
            self.issue_water_stress_alert(
                affected_regions=self.identify_stressed_regions(enhanced_state),
                recommendations=self.generate_mitigation_recommendations()
            )

        return enhanced_state, indicators
```

---

## 7. Continuous Learning and Feedback Loop

### 7.1 Model Improvement Workflow

```python
class ContinuousLearningWorkflow:
    def __init__(self):
        self.feedback_collectors = {
            "forecast_accuracy": ForecastAccuracyCollector(),
            "advisory_usefulness": AdvisoryFeedbackCollector(),
            "research_quality": ResearchQualityCollector(),
            "user_satisfaction": UserSatisfactionCollector()
        }
        self.retraining_pipeline = RetrainingPipeline()
        self.a_b_testing = ABTestingFramework()

    def weekly_learning_cycle(self):
        # Step 1: Collect feedback from all channels
        feedback = {}
        for name, collector in self.feedback_collectors.items():
            feedback[name] = collector.collect(period="7d")

        # Step 2: Analyze performance degradation
        degradation_report = self.analyze_degradation(feedback)

        # Step 3: Identify models needing retraining
        models_to_retrain = []
        for model_name, metrics in degradation_report.items():
            if metrics["accuracy_drop"] > 0.05:  # 5% accuracy drop
                models_to_retrain.append(model_name)

        # Step 4: Prepare new training data
        for model_name in models_to_retrain:
            new_data = self.prepare_training_data(
                model_name=model_name,
                include_recent_observations=True,
                include_user_feedback=True,
                include_new_research=True
            )

            # Step 5: Retrain with A/B testing
            new_model = self.retraining_pipeline.retrain(
                model_name=model_name,
                data=new_data,
                hyperparameters=self.optimize_hyperparameters(model_name)
            )

            # Step 6: Evaluate against current production model
            evaluation = self.a_b_testing.evaluate(
                new_model=new_model,
                baseline_model=self.get_production_model(model_name),
                test_data=self.holdout_test_set
            )

            # Step 7: Deploy if improved
            if evaluation.improvement > 0.02:  # 2% improvement
                self.deploy_model(
                    model_name=model_name,
                    model=new_model,
                    strategy="canary",  # Gradual rollout
                    rollback_plan=True
                )

                # Step 8: Update model card
                self.update_model_card(
                    model_name=model_name,
                    new_metrics=evaluation.metrics,
                    training_data_summary=new_data.summary,
                    changes=self.generate_changelog()
                )

        # Step 9: Generate learning report
        report = self.generate_learning_report(
            feedback=feedback,
            degradation=degradation_report,
            retrained_models=models_to_retrain,
            deployments=successful_deployments
        )

        # Step 10: Share with stakeholders
        self.share_report(report, recipients=["AI_Team", "Domain_Experts", "Management"])
```

### 7.2 Human Feedback Integration

```python
class HumanFeedbackIntegration:
    def __init__(self):
        self.expert_review_queue = ReviewQueue()
        self.feedback_embedding = FeedbackEmbeddingModel()
        self.model_updater = InContextLearningUpdater()

    def process_expert_feedback(self, feedback: ExpertFeedback):
        # Step 1: Categorize feedback
        category = self.categorize_feedback(feedback)
        # Categories: FACTUAL_CORRECTION, METHODOLOGY, INTERPRETATION, POLICY, ETHICS

        # Step 2: Embed feedback for semantic search
        embedding = self.feedback_embedding.encode(feedback.text)

        # Step 3: Find related model predictions
        related_predictions = self.find_related_predictions(
            embedding=embedding,
            model=feedback.target_model,
            time_window="30d"
        )

        # Step 4: Determine action type
        if category == "FACTUAL_CORRECTION":
            # Immediate correction to knowledge base
            self.correct_knowledge_base(
                correction=feedback.correction,
                source=feedback.expert_credentials,
                evidence=feedback.supporting_evidence
            )

        elif category == "METHODOLOGY":
            # Flag for model retraining with new approach
            self.flag_for_retraining(
                model=feedback.target_model,
                issue=feedback.description,
                priority="high"
            )

        elif category == "INTERPRETATION":
            # Update model's explanation generation
            self.update_explanation_patterns(
                model=feedback.target_model,
                pattern=feedback.suggested_interpretation
            )

        elif category == "POLICY":
            # Update policy recommendation templates
            self.update_policy_templates(
                domain=feedback.domain,
                recommendation=feedback.suggested_recommendation
            )

        elif category == "ETHICS":
            # Escalate to ethics board
            self.escalate_to_ethics_board(feedback)

        # Step 5: Log feedback for audit trail
        self.log_feedback(
            feedback=feedback,
            action_taken=action,
            timestamp=datetime.now(),
            expert_id=feedback.expert_id
        )

        # Step 6: Thank expert and close loop
        self.send_acknowledgment(feedback.expert_id, action)
```

---

## 8. Ethical AI and Fairness Workflow

### 8.1 Bias Detection and Mitigation

```python
class EthicalAIWorkflow:
    def __init__(self):
        self.fairness_metrics = FairnessMetricsLibrary()
        self.bias_detector = BiasDetector()
        self.mitigation_strategies = MitigationStrategies()
        self.ethics_board = EthicsReviewBoard()

    def pre_deployment_audit(self, model: AIModel, use_case: str):
        # Step 1: Demographic parity check
        demographic_parity = self.fairness_metrics.demographic_parity(
            model=model,
            protected_attributes=["region", "income_level", "language"],
            threshold=0.05
        )

        # Step 2: Equal opportunity check
        equal_opportunity = self.fairness_metrics.equal_opportunity(
            model=model,
            protected_attributes=["region", "income_level", "language"],
            threshold=0.05
        )

        # Step 3: Calibration check
        calibration = self.fairness_metrics.calibration(
            model=model,
            protected_attributes=["region", "income_level", "language"],
            threshold=0.05
        )

        # Step 4: Bias detection across subgroups
        subgroup_bias = self.bias_detector.detect(
            model=model,
            subgroups=self.define_subgroups(),
            min_sample_size=100
        )

        # Step 5: Generate fairness report
        fairness_report = self.generate_fairness_report(
            demographic_parity=demographic_parity,
            equal_opportunity=equal_opportunity,
            calibration=calibration,
            subgroup_bias=subgroup_bias
        )

        # Step 6: Apply mitigation if needed
        if fairness_report.has_violations:
            mitigated_model = self.mitigation_strategies.apply(
                model=model,
                violations=fairness_report.violations,
                strategy="reweighting"  # or "adversarial_debiasing", "fair_repr_learning"
            )

            # Re-evaluate after mitigation
            post_mitigation_report = self.pre_deployment_audit(
                model=mitigated_model,
                use_case=use_case
            )

            if post_mitigation_report.has_violations:
                # Escalate to ethics board
                self.ethics_board.review(
                    model=mitigated_model,
                    report=post_mitigation_report,
                    use_case=use_case
                )
                return None  # Deployment blocked pending review

            model = mitigated_model

        # Step 7: Ethics board approval for high-stakes use cases
        if use_case in ["DISASTER_RESPONSE", "POLICY_RECOMMENDATION"]:
            approval = self.ethics_board.approve(
                model=model,
                fairness_report=fairness_report,
                use_case=use_case
            )
            if not approval:
                return None

        return model

    def define_subgroups(self) -> List[Subgroup]:
        return [
            Subgroup(name="Smallholder_Farmers", criteria="land_area < 2_hectares"),
            Subgroup(name="Marginalized_Regions", criteria="district_in_development_index < 0.5"),
            Subgroup(name="Non_Hindi_Speakers", criteria="primary_language != 'hindi'"),
            Subgroup(name="Coastal_Vulnerable", criteria="distance_to_coast < 50km"),
            Subgroup(name="Tribal_Communities", criteria="tribal_population_pct > 0.3"),
            Subgroup(name="Women_Farmers", criteria="gender == 'female' AND occupation == 'farmer'"),
        ]
```

### 8.2 Explainability Workflow

```python
class ExplainabilityWorkflow:
    def __init__(self):
        self.shap_explainer = SHAPTreeExplainer()
        self.lime_explainer = LIMEExplainer()
        self.concept_explainer = ConceptExplainer()
        self.nlg_generator = NLGExplanationGenerator()

    def explain_prediction(self, prediction: Prediction, audience: str):
        # Step 1: Feature importance (SHAP)
        shap_values = self.shap_explainer.explain(
            model=prediction.model,
            input=prediction.input_features
        )

        # Step 2: Local explanation (LIME)
        lime_explanation = self.lime_explainer.explain(
            model=prediction.model,
            input=prediction.input_features,
            num_features=10
        )

        # Step 3: Concept-level explanation
        concept_explanation = self.concept_explainer.explain(
            prediction=prediction,
            domain_knowledge=self.climate_knowledge_base
        )

        # Step 4: Generate natural language explanation
        if audience == "SCIENTIST":
            explanation = self.nlg_generator.generate_scientific(
                shap=shap_values,
                lime=lime_explanation,
                concepts=concept_explanation,
                confidence=prediction.confidence,
                uncertainty=prediction.uncertainty
            )
        elif audience == "POLICYMAKER":
            explanation = self.nlg_generator.generate_policy(
                shap=shap_values,
                concepts=concept_explanation,
                recommendations=prediction.recommendations,
                risks=prediction.risks
            )
        elif audience == "FARMER":
            explanation = self.nlg_generator.generate_simple(
                key_factors=shap_values.top_3,
                action=prediction.recommended_action,
                confidence=prediction.confidence,
                language=prediction.user_language
            )
        elif audience == "PUBLIC":
            explanation = self.nlg_generator.generate_infographic(
                prediction=prediction,
                visual_elements=self.generate_visuals(prediction)
            )

        # Step 5: Attach to prediction
        prediction.explanation = explanation
        prediction.explanation_metadata = {
            "shap_values": shap_values,
            "lime_weights": lime_explanation,
            "concepts": concept_explanation,
            "confidence": prediction.confidence,
            "model_version": prediction.model.version
        }

        return prediction
```

---

## 9. Multi-Agent Collaboration Protocol

### 9.1 Agent Communication Protocol

```python
class AgentCommunicationProtocol:
    def __init__(self):
        self.message_bus = RedisMessageBus()
        self.schema_registry = AvroSchemaRegistry()

    def send_message(self, sender: str, recipient: str, message: AgentMessage):
        # Step 1: Validate message against schema
        schema = self.schema_registry.get_schema(message.message_type)
        if not self.validate(message, schema):
            raise ValidationError("Message does not conform to schema")

        # Step 2: Add metadata
        enriched_message = EnrichedMessage(
            original=message,
            timestamp=datetime.now(),
            trace_id=message.trace_id or uuid4(),
            sender_agent=sender,
            recipient_agent=recipient,
            priority=message.priority or "NORMAL",
            ttl=message.ttl or 3600  # 1 hour default
        )

        # Step 3: Publish to message bus
        self.message_bus.publish(
            channel=f"agent.{recipient}.inbox",
            message=enriched_message
        )

        # Step 4: Log for observability
        self.log_message(enriched_message)

    def receive_message(self, agent: str, timeout: int = 30) -> Optional[EnrichedMessage]:
        message = self.message_bus.pop(
            channel=f"agent.{agent}.inbox",
            timeout=timeout
        )

        if message:
            self.acknowledge_message(message)
            return message

        return None

    def broadcast(self, sender: str, message: AgentMessage, agent_type: str = None):
        recipients = self.registry.get_agents(agent_type=agent_type)
        for recipient in recipients:
            self.send_message(sender, recipient, message)
```

### 9.2 Consensus Mechanism

```python
class ConsensusMechanism:
    def __init__(self):
        self.agents = {
            "physics_model": PhysicsModelAgent(),
            "data_model": DataDrivenAgent(),
            "expert_model": ExpertSystemAgent(),
            "ensemble_model": EnsembleAgent()
        }
        self.consensus_threshold = 0.75

    def reach_consensus(self, decision_context: DecisionContext) -> ConsensusResult:
        # Step 1: Collect predictions from all agents
        predictions = {}
        for name, agent in self.agents.items():
            predictions[name] = agent.predict(decision_context)

        # Step 2: Calculate agreement score
        agreement = self.calculate_agreement(predictions)

        # Step 3: If high agreement, return consensus
        if agreement.score >= self.consensus_threshold:
            return ConsensusResult(
                decision=self.aggregate_predictions(predictions),
                confidence=agreement.score,
                dissenting_views=agreement.dissenters,
                explanation="High agreement across all models"
            )

        # Step 4: If low agreement, investigate reasons
        disagreement_analysis = self.analyze_disagreement(predictions)

        # Step 5: Request additional data if needed
        if disagreement_analysis.cause == "INSUFFICIENT_DATA":
            additional_data = self.request_additional_data(
                decision_context,
                disagreement_analysis.data_gaps
            )
            return self.reach_consensus(
                decision_context.with_data(additional_data)
            )

        # Step 6: If fundamental disagreement, escalate to human
        if disagreement_analysis.cause == "FUNDAMENTAL_UNCERTAINTY":
            return ConsensusResult(
                decision=None,
                confidence=agreement.score,
                dissenting_views=agreement.dissenters,
                explanation="Fundamental uncertainty - human decision required",
                escalation_required=True,
                escalation_target="Chief_Scientist"
            )

        # Step 7: Weighted consensus (physics model gets higher weight)
        weighted_decision = self.weighted_consensus(
            predictions,
            weights={
                "physics_model": 0.4,
                "data_model": 0.3,
                "expert_model": 0.2,
                "ensemble_model": 0.1
            }
        )

        return ConsensusResult(
            decision=weighted_decision,
            confidence=agreement.score,
            dissenting_views=agreement.dissenters,
            explanation="Weighted consensus due to partial disagreement"
        )
```

---

## 10. Performance and Cost Optimization

### 10.1 Intelligent Resource Allocation

```python
class ResourceOptimizer:
    def __init__(self):
        self.cost_model = CloudCostModel()
        self.priority_queue = PriorityTaskQueue()
        self.spot_instance_manager = SpotInstanceManager()

    def optimize(self, task_queue: List[AITask]):
        # Step 1: Classify tasks by urgency and compute needs
        classified_tasks = []
        for task in task_queue:
            priority = self.assess_priority(task)
            compute_need = self.estimate_compute(task)
            classified_tasks.append((task, priority, compute_need))

        # Step 2: Allocate resources
        for task, priority, compute in classified_tasks:
            if priority == "CRITICAL":
                self.allocate_on_demand(task, compute)
            elif priority == "HIGH":
                if self.has_reserved_capacity(compute):
                    self.allocate_reserved(task, compute)
                else:
                    self.allocate_on_demand(task, compute)
            elif priority == "NORMAL":
                self.spot_instance_manager.submit(task, compute)
            elif priority == "LOW":
                self.schedule_off_peak(task, compute)

        # Step 3: Model compression for inference
        for model in self.get_inference_models():
            if model.latency_p99 > self.sla_threshold:
                compressed = self.compress_model(model)
                self.deploy_compressed(compressed, canary=True)

        # Step 4: Cache frequently used embeddings
        self.update_embedding_cache(
            hot_embeddings=self.get_hot_embeddings(),
            cache_size="50GB"
        )
```

### 10.2 Cost Tracking per Research Query

```python
class CostTracker:
    def __init__(self):
        self.token_usage = TokenUsageTracker()
        self.api_costs = APICostTracker()
        self.compute_costs = ComputeCostTracker()

    def track_research_query(self, query_id: str):
        costs = {
            "llm_tokens": {
                "planner": self.token_usage.get(query_id, "planner"),
                "researchers": self.token_usage.get(query_id, "researchers"),
                "writer": self.token_usage.get(query_id, "writer"),
                "total_tokens": sum(...),
                "estimated_cost_usd": self.calculate_llm_cost(...)
            },
            "search_apis": {
                "google_search": self.api_costs.get(query_id, "google"),
                "scholar_api": self.api_costs.get(query_id, "scholar"),
                "exa_api": self.api_costs.get(query_id, "exa"),
                "total_cost_usd": sum(...)
            },
            "compute": {
                "embedding_generation": self.compute_costs.get(query_id, "embeddings"),
                "reranking": self.compute_costs.get(query_id, "reranking"),
                "chunking": self.compute_costs.get(query_id, "chunking"),
                "total_cost_usd": sum(...)
            },
            "total_cost_usd": 0.0,
            "cost_per_citation": 0.0,
            "cost_per_1000_words": 0.0
        }

        costs["total_cost_usd"] = (
            costs["llm_tokens"]["estimated_cost_usd"] +
            costs["search_apis"]["total_cost_usd"] +
            costs["compute"]["total_cost_usd"]
        )

        # Alert if cost exceeds budget
        if costs["total_cost_usd"] > self.budget_per_query:
            self.alert_budget_exceeded(query_id, costs)

        return costs
```

---

## 11. Emergency Protocols

### 11.1 Extreme Event Response Workflow

```python
class ExtremeEventResponse:
    def __init__(self):
        self.event_detector = ExtremeEventDetector()
        self.escalation_matrix = EscalationMatrix()
        self.communication_hub = EmergencyCommunicationHub()

    def monitor_and_respond(self):
        while True:
            # Step 1: Detect extreme events
            events = self.event_detector.detect(
                forecasts=self.get_all_forecasts(),
                observations=self.get_all_observations(),
                thresholds=self.extreme_event_thresholds
            )

            for event in events:
                # Step 2: Classify severity
                severity = self.classify_severity(event)

                # Step 3: Determine response level
                if severity == "CATASTROPHIC":
                    self.activate_level_1_response(event)
                elif severity == "SEVERE":
                    self.activate_level_2_response(event)
                elif severity == "MODERATE":
                    self.activate_level_3_response(event)
                elif severity == "MINOR":
                    self.activate_level_4_response(event)

                # Step 4: Log and learn
                self.log_event_response(event, severity, response)

            time.sleep(60)  # Check every minute during active season

    def activate_level_1_response(self, event: ExtremeEvent):
        # Immediate actions (within 5 minutes)
        self.communication_hub.alert(
            recipients=["PMO", "NDMA", "State_CMs", "IMD_DG"],
            channel="SECURE_HOTLINE",
            message=event.summary
        )

        # Auto-generate situation report
        sitrep = self.generate_situation_report(event)
        self.distribute_sitrep(sitrep, classification="RESTRICTED")

        # Activate all simulation resources
        self.prioritize_hpc_resources(event)

        # Update all models with real-time data
        self.trigger_emergency_data_assimilation(event)

        # Generate ensemble forecasts with maximum resolution
        self.run_high_resolution_ensemble(event)

        # Issue public alerts
        self.issue_national_alert(event)

        # Coordinate with international agencies
        self.notify_wmo_and_un(event)
```

---

## 12. Appendices

### Appendix A: Prompt Library

A comprehensive library of prompts for all AI agents, organized by:
- Agent type (Planner, Researcher, Writer, etc.)
- Domain (Atmospheric, Oceanic, Agricultural, Urban, Hydrological)
- Audience (Scientist, Policymaker, Farmer, Public)
- Language (English, Hindi, Tamil, Bengali, Marathi, Telugu, Kannada, Gujarati, Malayalam)
- Output format (Markdown, LaTeX, JSON, HTML, Voice)

### Appendix B: Model Cards

Standardized model cards for all deployed AI models, including:
- Model name, version, and type
- Intended use and limitations
- Training data summary and provenance
- Performance metrics across subgroups
- Fairness audit results
- Environmental impact (carbon footprint)
- Known failure modes and mitigation strategies
- Contact information for feedback

### Appendix C: Feedback Forms

Structured feedback collection forms for:
- Forecast accuracy (binary correct/incorrect + severity rating)
- Advisory usefulness (Likert scale 1-5 + qualitative feedback)
- Research quality (citation accuracy, comprehensiveness, relevance)
- User satisfaction (NPS score + feature requests)
- Ethical concerns (anonymous reporting channel)

### Appendix D: Glossary of AI Terms

- **Agent:** Autonomous AI system with specific role and capabilities
- **DAG:** Directed Acyclic Graph - research plan structure
- **Ensemble:** Multiple model predictions combined for robustness
- **HITL:** Human-in-the-Loop - human approval at critical decision points
- **PINN:** Physics-Informed Neural Network
- **SHAP:** SHapley Additive exPlanations - feature importance method
- **LIME:** Local Interpretable Model-agnostic Explanations
- **MCP:** Model Context Protocol - tool use framework
- **RAG:** Retrieval-Augmented Generation
- **ReAct:** Reasoning + Acting loop for agent decision-making

---

*Document Owner: AI Governance Board, NCCC*
*Review Cycle: Weekly during development; Monthly in production*
*Next Review: 2026-07-05*
