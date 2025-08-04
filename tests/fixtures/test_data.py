"""
Comprehensive test data and fixtures for figure/table extraction tests.

This module provides reusable test data including mock PDF content, XML samples,
and various document structures for testing the enhanced extraction functionality.
"""

import pytest
from typing import Dict, List, Any
from unittest.mock import Mock


class MockTestData:
    """Mock test data provider for consistent testing across modules."""
    
    @staticmethod
    def get_sample_pdf_content() -> Dict[str, Any]:
        """Get comprehensive sample PDF content for testing."""
        return {
            "text": """
            Novel Therapeutic Approaches in Cardiovascular Medicine: A Randomized Controlled Trial
            
            Abstract
            Background: Current therapeutic approaches in cardiovascular medicine show limited efficacy in certain patient populations.
            Objectives: To evaluate the safety and efficacy of a novel therapeutic intervention compared to standard care.
            Methods: Double-blind, randomized controlled trial with 150 participants followed for 24 weeks.
            Results: Significant improvements in primary outcomes were observed (p<0.001) with acceptable safety profile.
            Conclusions: The novel therapeutic approach demonstrates superior efficacy and safety compared to standard care.
            
            1. Introduction
            Cardiovascular disease remains the leading cause of morbidity and mortality worldwide, affecting millions of patients annually.
            Despite advances in therapeutic interventions, significant gaps remain in treatment efficacy, particularly in high-risk populations.
            Previous studies have demonstrated limitations in current approaches, necessitating the development of novel therapeutic strategies.
            The pathophysiology underlying cardiovascular disease involves complex interactions between genetic, environmental, and lifestyle factors.
            
            2. Methods
            
            2.1 Study Design and Participants
            This was a multicenter, double-blind, randomized controlled trial conducted at five academic medical centers between January 2022 and December 2023.
            The study protocol was approved by institutional review boards at all participating sites and registered with ClinicalTrials.gov.
            
            Figure 1. CONSORT flow diagram showing participant enrollment, randomization, and follow-up procedures.
            A total of 250 patients were screened for eligibility, with 150 meeting inclusion criteria and providing informed consent.
            Participants were randomly assigned to either the intervention group (n=75) or control group (n=75) using computer-generated randomization.
            The randomization was stratified by age (<65 vs ≥65 years), gender, and presence of diabetes mellitus.
            Follow-up assessments were conducted at baseline, 4, 8, 12, 16, 20, and 24 weeks, with a final safety follow-up at 28 weeks.
            
            2.2 Baseline Characteristics
            Table 1. Baseline demographic and clinical characteristics of study participants by treatment group.
            
            Characteristic                    | Control Group (n=75) | Intervention Group (n=75) | P-value
            Age, years (mean ± SD)           | 58.3 ± 12.7          | 59.1 ± 11.9               | 0.712
            Male gender, n (%)               | 42 (56.0)            | 46 (61.3)                 | 0.502
            Race/ethnicity, n (%)            |                      |                           | 0.823
              White                          | 52 (69.3)            | 54 (72.0)                 |
              Black or African American      | 15 (20.0)            | 13 (17.3)                 |
              Hispanic or Latino             | 6 (8.0)              | 6 (8.0)                   |
              Other                          | 2 (2.7)              | 2 (2.7)                   |
            Body mass index, kg/m² (mean ± SD) | 28.4 ± 4.2        | 27.9 ± 4.1                | 0.486
            Systolic blood pressure, mmHg    | 142.3 ± 18.2         | 144.1 ± 19.7              | 0.567
            Diastolic blood pressure, mmHg   | 87.2 ± 12.4          | 88.7 ± 13.1               | 0.481
            Heart rate, bpm                  | 74.2 ± 11.8          | 73.6 ± 12.3               | 0.762
            
            Medical history, n (%)
            Diabetes mellitus                | 28 (37.3)            | 32 (42.7)                 | 0.513
            Hypertension                     | 56 (74.7)            | 58 (77.3)                 | 0.721
            Hyperlipidemia                   | 48 (64.0)            | 51 (68.0)                 | 0.624
            Current smoking                  | 18 (24.0)            | 21 (28.0)                 | 0.592
            Family history of CVD            | 34 (45.3)            | 37 (49.3)                 | 0.636
            
            Laboratory values (mean ± SD)
            Total cholesterol, mg/dL         | 189.4 ± 32.7         | 192.8 ± 35.1              | 0.543
            LDL cholesterol, mg/dL           | 112.3 ± 28.9         | 115.7 ± 31.2              | 0.486
            HDL cholesterol, mg/dL           | 42.1 ± 8.7           | 41.8 ± 9.2                | 0.834
            Triglycerides, mg/dL             | 156.8 ± 42.3         | 162.4 ± 45.7              | 0.442
            Fasting glucose, mg/dL           | 108.7 ± 24.1         | 111.2 ± 26.8              | 0.561
            HbA1c, %                         | 6.8 ± 1.2            | 6.9 ± 1.3                 | 0.623
            Creatinine, mg/dL                | 1.12 ± 0.23          | 1.08 ± 0.21               | 0.278
            
            Baseline characteristics were well-balanced between treatment groups, with no statistically significant differences observed.
            The study population was representative of the target patient demographic for cardiovascular interventions.
            
            2.3 Intervention Protocol
            Figure 2. Detailed intervention protocol timeline showing three distinct treatment phases over the 24-week study period.
            
            Phase I (Weeks 1-4): Initial Assessment and Dose Escalation
            - Comprehensive baseline assessment including medical history, physical examination, and laboratory studies
            - Initiation of study medication with careful dose titration based on patient response and tolerability
            - Weekly monitoring visits with safety assessments and dose adjustments as needed
            - Patient education regarding medication adherence, lifestyle modifications, and recognition of adverse effects
            
            Phase II (Weeks 5-16): Maintenance Therapy and Optimization
            - Continuation of optimized study medication dosing established in Phase I
            - Bi-weekly monitoring visits with clinical assessments and laboratory monitoring
            - Concurrent lifestyle interventions including dietary counseling and exercise prescription
            - Continuous safety monitoring with predefined stopping rules for adverse events
            
            Phase III (Weeks 17-24): Long-term Efficacy and Safety Assessment
            - Maintenance of therapeutic dosing with monthly monitoring visits
            - Comprehensive efficacy assessments using validated clinical outcome measures
            - Long-term safety evaluation including cardiovascular events and laboratory abnormalities
            - Preparation for study completion and transition to standard care or extension study
            
            The intervention protocol was designed to maximize therapeutic benefit while ensuring patient safety throughout the study period.
            All protocol deviations were documented and reviewed by the data safety monitoring board.
            
            2.4 Outcome Measures
            The primary efficacy endpoint was the change from baseline in cardiovascular risk score at 24 weeks, measured using a validated composite assessment tool.
            Secondary endpoints included individual components of the risk score, quality of life measures, biomarker changes, and safety parameters.
            
            2.5 Statistical Analysis
            All efficacy analyses were performed using the intention-to-treat principle, including all randomized participants who received at least one dose of study medication.
            Safety analyses included all participants who received any amount of study treatment.
            Continuous variables were analyzed using analysis of covariance (ANCOVA) with baseline values as covariates.
            Categorical variables were compared using chi-square tests or Fisher's exact test as appropriate.
            A p-value of <0.05 was considered statistically significant for all analyses.
            
            3. Results
            
            3.1 Participant Flow and Disposition
            Of the 250 patients screened, 150 met eligibility criteria and were randomized to treatment.
            The most common reasons for screen failure were failure to meet inclusion criteria (n=62), withdrawal of consent (n=28), and investigator discretion (n=10).
            Study completion rates were high, with 142 participants (94.7%) completing the 24-week treatment period.
            Early discontinuation was primarily due to adverse events (n=5), loss to follow-up (n=2), and withdrawal of consent (n=1).
            
            3.2 Primary Efficacy Outcomes
            Figure 3. Primary efficacy results showing cardiovascular risk score changes from baseline to 24 weeks by treatment group.
            
            The primary endpoint analysis demonstrated significant superiority of the intervention compared to control.
            Mean change in cardiovascular risk score from baseline was -12.4 ± 6.8 points in the intervention group compared to -4.2 ± 5.9 points in the control group.
            The between-group difference was -8.2 points (95% CI: -10.4 to -6.0; p<0.001), representing a clinically meaningful improvement.
            
            Box plots in Figure 3 illustrate the distribution of risk score changes, with the intervention group showing consistently greater improvements across all quartiles.
            Statistical significance was maintained across all prespecified sensitivity analyses and remained robust after adjustment for baseline covariates.
            
            Table 2. Primary and secondary efficacy outcomes at 24 weeks by treatment group.
            
            Outcome Measure                           | Control Group | Intervention Group | Between-Group Difference | P-value
                                                     | (n=75)        | (n=75)             | (95% CI)                 |
            Primary Endpoint
            Cardiovascular risk score change          | -4.2 ± 5.9    | -12.4 ± 6.8        | -8.2 (-10.4 to -6.0)     | <0.001
            
            Secondary Endpoints
            Systolic blood pressure change, mmHg     | -3.1 ± 8.7    | -11.8 ± 9.4        | -8.7 (-11.9 to -5.5)     | <0.001
            Diastolic blood pressure change, mmHg    | -1.8 ± 6.2    | -7.2 ± 7.1         | -5.4 (-7.6 to -3.2)      | <0.001
            Heart rate change, bpm                   | -0.8 ± 4.3    | -5.2 ± 5.1         | -4.4 (-6.0 to -2.8)      | <0.001
            Total cholesterol change, mg/dL          | -8.2 ± 18.4   | -24.7 ± 21.3       | -16.5 (-22.8 to -10.2)   | <0.001
            LDL cholesterol change, mg/dL            | -6.4 ± 15.2   | -19.8 ± 17.9       | -13.4 (-18.7 to -8.1)    | <0.001
            HDL cholesterol change, mg/dL            | 1.2 ± 4.8     | 4.7 ± 5.9          | 3.5 (1.8 to 5.2)         | <0.001
            Triglycerides change, mg/dL              | -12.3 ± 28.7  | -38.4 ± 34.2       | -26.1 (-36.2 to -16.0)   | <0.001
            Fasting glucose change, mg/dL            | -2.1 ± 12.4   | -9.8 ± 14.7        | -7.7 (-12.1 to -3.3)     | 0.001
            HbA1c change, %                          | -0.1 ± 0.4    | -0.5 ± 0.6         | -0.4 (-0.6 to -0.2)      | <0.001
            
            Quality of Life Measures
            SF-36 Physical Component Score           | 2.3 ± 6.8     | 8.7 ± 7.9          | 6.4 (4.1 to 8.7)         | <0.001
            SF-36 Mental Component Score             | 1.8 ± 5.9     | 6.2 ± 7.1          | 4.4 (2.3 to 6.5)         | <0.001
            Disease-specific quality of life score   | 3.1 ± 4.2     | 9.4 ± 5.8          | 6.3 (4.6 to 8.0)         | <0.001
            
            All secondary endpoints demonstrated statistically significant improvements favoring the intervention group.
            Effect sizes were generally large and clinically meaningful across all measured parameters.
            
            3.3 Subgroup Analyses
            Figure 4. Forest plot showing treatment effects across predefined subgroups with hazard ratios and 95% confidence intervals.
            
            Subgroup analyses were performed to evaluate treatment effects across different patient populations.
            Consistent benefits were observed across all predefined subgroups, with no evidence of heterogeneity of treatment effect.
            
            Table 3. Subgroup analysis of primary efficacy endpoint by baseline patient characteristics.
            
            Subgroup                        | Control Group | Intervention Group | Treatment Effect | P-interaction
                                           | Mean Change   | Mean Change        | (95% CI)         |
            Overall Population             | -4.2 ± 5.9    | -12.4 ± 6.8        | -8.2 (-10.4 to -6.0) | —
            
            Age Group
            <65 years (n=89)               | -4.8 ± 6.2    | -13.1 ± 7.1        | -8.3 (-11.2 to -5.4) | 0.892
            ≥65 years (n=61)               | -3.4 ± 5.4    | -11.2 ± 6.2        | -7.8 (-11.4 to -4.2) |
            
            Gender
            Male (n=88)                    | -4.6 ± 6.1    | -12.8 ± 7.3        | -8.2 (-11.1 to -5.3) | 0.978
            Female (n=62)                  | -3.7 ± 5.6    | -11.8 ± 5.9        | -8.1 (-11.8 to -4.4) |
            
            Diabetes Status
            Diabetic (n=60)                | -3.8 ± 5.2    | -11.9 ± 6.4        | -8.1 (-11.3 to -4.9) | 0.876
            Non-diabetic (n=90)            | -4.5 ± 6.3    | -12.7 ± 7.1        | -8.2 (-11.7 to -4.7) |
            
            BMI Category
            <30 kg/m² (n=92)               | -4.4 ± 6.0    | -12.1 ± 6.5        | -7.7 (-10.7 to -4.7) | 0.643
            ≥30 kg/m² (n=58)               | -3.9 ± 5.7    | -12.9 ± 7.3        | -9.0 (-12.8 to -5.2) |
            
            Baseline Risk Score
            Low risk (<15 points) (n=48)    | -2.8 ± 4.1    | -8.9 ± 5.2         | -6.1 (-8.6 to -3.6)  | 0.234
            Moderate risk (15-25 points) (n=67) | -4.1 ± 5.8 | -12.3 ± 6.9       | -8.2 (-11.3 to -5.1) |
            High risk (>25 points) (n=35)   | -6.2 ± 7.4    | -16.8 ± 8.1        | -10.6 (-15.4 to -5.8)|
            
            The consistency of treatment effects across subgroups supports the generalizability of study findings to the broader target population.
            No significant interactions were observed, suggesting uniform benefit across patient characteristics.
            
            3.4 Safety and Tolerability
            Table 4. Summary of adverse events and safety parameters by treatment group during the 24-week study period.
            
            Safety Parameter                      | Control Group | Intervention Group | Risk Ratio      | P-value
                                                 | n (%)         | n (%)              | (95% CI)        |
            Any adverse event                    | 52 (69.3)     | 48 (64.0)          | 0.92 (0.73-1.16) | 0.489
            Serious adverse events               | 8 (10.7)      | 6 (8.0)            | 0.75 (0.28-2.01) | 0.571
            Adverse events leading to discontinuation | 3 (4.0)   | 2 (2.7)            | 0.67 (0.11-3.93) | 0.653
            
            Most Common Adverse Events (≥5% in either group)
            Headache                            | 18 (24.0)     | 15 (20.0)          | 0.83 (0.46-1.50) | 0.542
            Nausea                              | 12 (16.0)     | 14 (18.7)          | 1.17 (0.58-2.35) | 0.668
            Dizziness                           | 11 (14.7)     | 8 (10.7)           | 0.73 (0.32-1.66) | 0.448
            Fatigue                             | 9 (12.0)      | 11 (14.7)          | 1.22 (0.54-2.78) | 0.634
            Upper respiratory tract infection    | 8 (10.7)      | 6 (8.0)            | 0.75 (0.28-2.01) | 0.571
            Diarrhea                            | 6 (8.0)       | 4 (5.3)            | 0.67 (0.20-2.23) | 0.513
            Back pain                           | 5 (6.7)       | 7 (9.3)            | 1.40 (0.47-4.15) | 0.544
            
            Laboratory Abnormalities
            Elevated liver enzymes (>3x ULN)     | 2 (2.7)       | 1 (1.3)            | 0.50 (0.05-5.37) | 0.561
            Elevated creatinine (>1.5x baseline) | 1 (1.3)       | 2 (2.7)            | 2.00 (0.19-21.6) | 0.561
            Significant electrolyte abnormalities | 3 (4.0)       | 2 (2.7)            | 0.67 (0.11-3.93) | 0.653
            
            Cardiovascular Events
            Myocardial infarction               | 1 (1.3)       | 0 (0.0)            | —               | 0.314
            Stroke                              | 0 (0.0)       | 1 (1.3)            | —               | 0.314
            Hospitalization for heart failure   | 2 (2.7)       | 1 (1.3)            | 0.50 (0.05-5.37) | 0.561
            
            The intervention was generally well-tolerated with an acceptable safety profile.
            No new safety signals were identified, and the overall adverse event rate was similar between groups.
            Most adverse events were mild to moderate in severity and resolved without intervention.
            
            4. Discussion
            
            4.1 Principal Findings
            This randomized controlled trial demonstrates that the novel therapeutic intervention provides significant clinical benefits compared to standard care in patients with cardiovascular disease.
            The primary efficacy endpoint showed a clinically meaningful improvement with strong statistical significance (p<0.001).
            Secondary endpoints consistently favored the intervention group, supporting the robustness of the primary findings.
            
            The magnitude and consistency of treatment effects observed in this study exceed those reported in previous trials of similar interventions.
            As shown in Figure 3, the intervention group demonstrated superior outcomes across all quartiles of response, indicating broad therapeutic benefit.
            The safety profile was acceptable and comparable to control, with no unexpected adverse events or safety signals identified.
            
            4.2 Clinical Significance
            The observed treatment effects represent clinically meaningful improvements that are likely to translate into improved patient outcomes.
            The comprehensive nature of benefits observed across multiple cardiovascular risk factors suggests a multifaceted mechanism of action.
            Quality of life improvements, as detailed in Table 2, indicate that benefits extend beyond traditional biomarker changes to patient-reported outcomes.
            
            4.3 Subgroup Considerations
            The consistency of treatment effects across predefined subgroups (Table 3, Figure 4) supports the generalizability of findings to diverse patient populations.
            No evidence of treatment effect heterogeneity was observed, suggesting uniform benefit regardless of baseline patient characteristics.
            These findings support the potential for broad clinical application of the intervention across the target patient population.
            
            4.4 Safety Considerations
            The safety profile observed in this study is consistent with the known safety profile of similar interventions and acceptable for the target indication.
            As shown in Table 4, adverse event rates were comparable between treatment groups, with no increase in serious adverse events.
            Long-term safety monitoring will be important as clinical use expands beyond the controlled trial setting.
            
            4.5 Study Limitations
            Several limitations should be acknowledged in interpreting these results.
            The relatively short follow-up period of 24 weeks may not capture long-term effects or rare adverse events.
            The study population was recruited from academic medical centers and may not be fully representative of the broader patient population.
            Additionally, the open-label extension phase results are not yet available and may provide additional insights into long-term efficacy and safety.
            
            4.6 Future Directions
            Future research should focus on longer-term follow-up to establish durability of treatment effects and comprehensive safety profile.
            Investigation of optimal dosing strategies and combination therapies may further enhance therapeutic benefits.
            Real-world evidence studies will be important to validate findings in broader clinical practice settings.
            
            5. Conclusions
            This randomized controlled trial provides robust evidence for the efficacy and safety of the novel therapeutic intervention in patients with cardiovascular disease.
            The intervention demonstrated statistically significant and clinically meaningful improvements in the primary efficacy endpoint and all secondary endpoints.
            The safety profile was acceptable and comparable to control, supporting a favorable benefit-risk profile.
            These findings support the clinical development and potential regulatory approval of this novel therapeutic approach.
            
            The consistent treatment effects observed across diverse patient subgroups and the magnitude of clinical benefits suggest this intervention may address an important unmet medical need in cardiovascular medicine.
            Implementation of this therapy in clinical practice has the potential to improve outcomes for patients with cardiovascular disease.
            """,
            "extraction_method": "pdfplumber",
            "pages": 15,
            "raw_content": Mock(),
            "metadata": {
                "title": "Novel Therapeutic Approaches in Cardiovascular Medicine: A Randomized Controlled Trial",
                "authors": ["Smith, J.A.", "Johnson, M.B.", "Williams, C.D.", "Brown, R.E."],
                "doi": "10.1000/test.2024.12345",
                "journal": "Journal of Cardiovascular Medicine",
                "publication_date": "2024",
                "pages": "125-142"
            }
        }
    
    @staticmethod
    def get_sample_xml_content() -> Dict[str, Any]:
        """Get comprehensive sample XML content for testing."""
        return {
            "xml_content": """<?xml version="1.0" encoding="UTF-8"?>
            <article xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                     xmlns:mml="http://www.w3.org/1998/Math/MathML"
                     xmlns:xlink="http://www.w3.org/1999/xlink"
                     article-type="research-article">
                <front>
                    <journal-meta>
                        <journal-id journal-id-type="nlm-ta">J Cardiovasc Med</journal-id>
                        <journal-title-group>
                            <journal-title>Journal of Cardiovascular Medicine</journal-title>
                        </journal-title-group>
                    </journal-meta>
                    <article-meta>
                        <article-id pub-id-type="doi">10.1000/test.2024.12345</article-id>
                        <title-group>
                            <article-title>Novel Therapeutic Approaches in Cardiovascular Medicine: A Randomized Controlled Trial</article-title>
                        </title-group>
                        <contrib-group>
                            <contrib contrib-type="author">
                                <name>
                                    <surname>Smith</surname>
                                    <given-names>John A</given-names>
                                </name>
                            </contrib>
                            <contrib contrib-type="author">
                                <name>
                                    <surname>Johnson</surname>
                                    <given-names>Mary B</given-names>
                                </name>
                            </contrib>
                        </contrib-group>
                        <pub-date pub-type="epub">
                            <year>2024</year>
                        </pub-date>
                    </article-meta>
                </front>
                
                <body>
                    <sec id="introduction">
                        <title>Introduction</title>
                        <p>Cardiovascular disease remains the leading cause of morbidity and mortality worldwide, affecting millions of patients annually. Despite advances in therapeutic interventions, significant gaps remain in treatment efficacy, particularly in high-risk populations.</p>
                        <p>Previous studies have demonstrated limitations in current approaches, necessitating the development of novel therapeutic strategies. The pathophysiology underlying cardiovascular disease involves complex interactions between genetic, environmental, and lifestyle factors.</p>
                    </sec>
                    
                    <sec id="methods">
                        <title>Methods</title>
                        
                        <sec id="study-design">
                            <title>Study Design and Participants</title>
                            <p>This was a multicenter, double-blind, randomized controlled trial conducted at five academic medical centers between January 2022 and December 2023. The study protocol was approved by institutional review boards at all participating sites.</p>
                            
                            <fig id="fig1" position="float">
                                <label>Figure 1</label>
                                <caption>
                                    <title>CONSORT Flow Diagram</title>
                                    <p>Participant enrollment, randomization, and follow-up procedures showing screening of 250 patients, randomization of 150 participants, and completion rates. The flow diagram illustrates the progression from initial screening through final analysis, including reasons for exclusion and dropout rates at each stage.</p>
                                </caption>
                                <graphic xlink:href="figure1_consort.tiff" mimetype="image" mime-subtype="tiff" position="float"/>
                                <alternatives>
                                    <graphic xlink:href="figure1_consort.png" mimetype="image" mime-subtype="png"/>
                                    <graphic xlink:href="figure1_consort.eps" mimetype="application" mime-subtype="postscript"/>
                                </alternatives>
                            </fig>
                        </sec>
                        
                        <sec id="baseline-characteristics">
                            <title>Baseline Characteristics</title>
                            <p>Baseline demographic and clinical characteristics were well-balanced between treatment groups, with no statistically significant differences observed. The study population was representative of the target patient demographic for cardiovascular interventions.</p>
                            
                            <table-wrap id="tab1" position="float">
                                <label>Table 1</label>
                                <caption>
                                    <title>Baseline Demographic and Clinical Characteristics</title>
                                    <p>Comprehensive comparison of baseline characteristics between control and intervention groups, including demographic variables, medical history, and laboratory parameters. Data presented as mean ± standard deviation for continuous variables and n (%) for categorical variables.</p>
                                </caption>
                                <table frame="hsides" rules="groups">
                                    <thead>
                                        <tr>
                                            <th align="left">Characteristic</th>
                                            <th align="center">Control Group<break/>(n=75)</th>
                                            <th align="center">Intervention Group<break/>(n=75)</th>
                                            <th align="center">P-value</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td colspan="4"><bold>Demographics</bold></td>
                                        </tr>
                                        <tr>
                                            <td>Age, years (mean ± SD)</td>
                                            <td align="center">58.3 ± 12.7</td>
                                            <td align="center">59.1 ± 11.9</td>
                                            <td align="center">0.712</td>
                                        </tr>
                                        <tr>
                                            <td>Male gender, n (%)</td>
                                            <td align="center">42 (56.0)</td>
                                            <td align="center">46 (61.3)</td>
                                            <td align="center">0.502</td>
                                        </tr>
                                        <tr>
                                            <td>Body mass index, kg/m² (mean ± SD)</td>
                                            <td align="center">28.4 ± 4.2</td>
                                            <td align="center">27.9 ± 4.1</td>
                                            <td align="center">0.486</td>
                                        </tr>
                                        <tr>
                                            <td colspan="4"><bold>Vital Signs</bold></td>
                                        </tr>
                                        <tr>
                                            <td>Systolic blood pressure, mmHg</td>
                                            <td align="center">142.3 ± 18.2</td>
                                            <td align="center">144.1 ± 19.7</td>
                                            <td align="center">0.567</td>
                                        </tr>
                                        <tr>
                                            <td>Diastolic blood pressure, mmHg</td>
                                            <td align="center">87.2 ± 12.4</td>
                                            <td align="center">88.7 ± 13.1</td>
                                            <td align="center">0.481</td>
                                        </tr>
                                        <tr>
                                            <td>Heart rate, bpm</td>
                                            <td align="center">74.2 ± 11.8</td>
                                            <td align="center">73.6 ± 12.3</td>
                                            <td align="center">0.762</td>
                                        </tr>
                                        <tr>
                                            <td colspan="4"><bold>Medical History, n (%)</bold></td>
                                        </tr>
                                        <tr>
                                            <td>Diabetes mellitus</td>
                                            <td align="center">28 (37.3)</td>
                                            <td align="center">32 (42.7)</td>
                                            <td align="center">0.513</td>
                                        </tr>
                                        <tr>
                                            <td>Hypertension</td>
                                            <td align="center">56 (74.7)</td>
                                            <td align="center">58 (77.3)</td>
                                            <td align="center">0.721</td>
                                        </tr>
                                        <tr>
                                            <td>Hyperlipidemia</td>
                                            <td align="center">48 (64.0)</td>
                                            <td align="center">51 (68.0)</td>
                                            <td align="center">0.624</td>
                                        </tr>
                                        <tr>
                                            <td colspan="4"><bold>Laboratory Values (mean ± SD)</bold></td>
                                        </tr>
                                        <tr>
                                            <td>Total cholesterol, mg/dL</td>
                                            <td align="center">189.4 ± 32.7</td>
                                            <td align="center">192.8 ± 35.1</td>
                                            <td align="center">0.543</td>
                                        </tr>
                                        <tr>
                                            <td>LDL cholesterol, mg/dL</td>
                                            <td align="center">112.3 ± 28.9</td>
                                            <td align="center">115.7 ± 31.2</td>
                                            <td align="center">0.486</td>
                                        </tr>
                                        <tr>
                                            <td>HDL cholesterol, mg/dL</td>
                                            <td align="center">42.1 ± 8.7</td>
                                            <td align="center">41.8 ± 9.2</td>
                                            <td align="center">0.834</td>
                                        </tr>
                                        <tr>
                                            <td>Fasting glucose, mg/dL</td>
                                            <td align="center">108.7 ± 24.1</td>
                                            <td align="center">111.2 ± 26.8</td>
                                            <td align="center">0.561</td>
                                        </tr>
                                    </tbody>
                                </table>
                                <table-wrap-foot>
                                    <fn>
                                        <p>Data are presented as mean ± standard deviation for continuous variables and n (%) for categorical variables. P-values calculated using independent t-test for continuous variables and chi-square test for categorical variables. SD = standard deviation; BMI = body mass index; LDL = low-density lipoprotein; HDL = high-density lipoprotein.</p>
                                    </fn>
                                </table-wrap-foot>
                            </table-wrap>
                        </sec>
                        
                        <sec id="intervention-protocol">
                            <title>Intervention Protocol</title>
                            <p>The intervention protocol was designed to maximize therapeutic benefit while ensuring patient safety throughout the study period. The treatment regimen consisted of three distinct phases over the 24-week study duration.</p>
                            
                            <fig id="fig2" position="float">
                                <label>Figure 2</label>
                                <caption>
                                    <title>Intervention Protocol Timeline</title>
                                    <p>Detailed timeline showing the three-phase intervention protocol over 24 weeks. Phase I (weeks 1-4): dose escalation and safety monitoring; Phase II (weeks 5-16): maintenance therapy optimization; Phase III (weeks 17-24): long-term efficacy assessment. The figure illustrates key milestones, assessment points, and safety monitoring procedures throughout the study period.</p>
                                </caption>
                                <graphic xlink:href="figure2_protocol.tiff" mimetype="image" mime-subtype="tiff"/>
                                <alternatives>
                                    <graphic xlink:href="figure2_protocol.png" mimetype="image" mime-subtype="png"/>
                                </alternatives>
                            </fig>
                        </sec>
                    </sec>
                    
                    <sec id="results">
                        <title>Results</title>
                        
                        <sec id="primary-outcomes">
                            <title>Primary Efficacy Outcomes</title>
                            <p>The primary endpoint analysis demonstrated significant superiority of the intervention compared to control. Mean change in cardiovascular risk score from baseline was -12.4 ± 6.8 points in the intervention group compared to -4.2 ± 5.9 points in the control group.</p>
                            
                            <fig id="fig3" position="float">
                                <label>Figure 3</label>
                                <caption>
                                    <title>Primary Efficacy Results</title>
                                    <p>Box plots showing cardiovascular risk score changes from baseline to 24 weeks by treatment group. The intervention group demonstrated significantly greater improvements compared to control (p&lt;0.001). Boxes represent interquartile ranges, whiskers show 95% confidence intervals, horizontal lines indicate medians, and individual points represent outliers. Statistical significance markers: ***p&lt;0.001.</p>
                                </caption>
                                <graphic xlink:href="figure3_efficacy.tiff" mimetype="image" mime-subtype="tiff"/>
                                <alternatives>
                                    <graphic xlink:href="figure3_efficacy.png" mimetype="image" mime-subtype="png"/>
                                </alternatives>
                            </fig>
                            
                            <table-wrap id="tab2" position="float">
                                <label>Table 2</label>
                                <caption>
                                    <title>Primary and Secondary Efficacy Outcomes</title>
                                    <p>Comprehensive analysis of primary and secondary efficacy endpoints at 24 weeks, showing treatment effects and statistical comparisons between groups. All values represent change from baseline unless otherwise specified. Between-group differences calculated using ANCOVA with baseline values as covariates.</p>
                                </caption>
                                <table frame="hsides" rules="groups">
                                    <thead>
                                        <tr>
                                            <th align="left">Outcome Measure</th>
                                            <th align="center">Control Group<break/>(n=75)</th>
                                            <th align="center">Intervention Group<break/>(n=75)</th>
                                            <th align="center">Between-Group Difference<break/>(95% CI)</th>
                                            <th align="center">P-value</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td colspan="5"><bold>Primary Endpoint</bold></td>
                                        </tr>
                                        <tr>
                                            <td>Cardiovascular risk score change</td>
                                            <td align="center">-4.2 ± 5.9</td>
                                            <td align="center">-12.4 ± 6.8</td>
                                            <td align="center">-8.2 (-10.4 to -6.0)</td>
                                            <td align="center">&lt;0.001</td>
                                        </tr>
                                        <tr>
                                            <td colspan="5"><bold>Secondary Endpoints</bold></td>
                                        </tr>
                                        <tr>
                                            <td>Systolic blood pressure change, mmHg</td>
                                            <td align="center">-3.1 ± 8.7</td>
                                            <td align="center">-11.8 ± 9.4</td>
                                            <td align="center">-8.7 (-11.9 to -5.5)</td>
                                            <td align="center">&lt;0.001</td>
                                        </tr>
                                        <tr>
                                            <td>Diastolic blood pressure change, mmHg</td>
                                            <td align="center">-1.8 ± 6.2</td>
                                            <td align="center">-7.2 ± 7.1</td>
                                            <td align="center">-5.4 (-7.6 to -3.2)</td>
                                            <td align="center">&lt;0.001</td>
                                        </tr>
                                        <tr>
                                            <td>Total cholesterol change, mg/dL</td>
                                            <td align="center">-8.2 ± 18.4</td>
                                            <td align="center">-24.7 ± 21.3</td>
                                            <td align="center">-16.5 (-22.8 to -10.2)</td>
                                            <td align="center">&lt;0.001</td>
                                        </tr>
                                        <tr>
                                            <td>LDL cholesterol change, mg/dL</td>
                                            <td align="center">-6.4 ± 15.2</td>
                                            <td align="center">-19.8 ± 17.9</td>
                                            <td align="center">-13.4 (-18.7 to -8.1)</td>
                                            <td align="center">&lt;0.001</td>
                                        </tr>
                                        <tr>
                                            <td>HDL cholesterol change, mg/dL</td>
                                            <td align="center">1.2 ± 4.8</td>
                                            <td align="center">4.7 ± 5.9</td>
                                            <td align="center">3.5 (1.8 to 5.2)</td>
                                            <td align="center">&lt;0.001</td>
                                        </tr>
                                        <tr>
                                            <td colspan="5"><bold>Quality of Life Measures</bold></td>
                                        </tr>
                                        <tr>
                                            <td>SF-36 Physical Component Score</td>
                                            <td align="center">2.3 ± 6.8</td>
                                            <td align="center">8.7 ± 7.9</td>
                                            <td align="center">6.4 (4.1 to 8.7)</td>
                                            <td align="center">&lt;0.001</td>
                                        </tr>
                                        <tr>
                                            <td>SF-36 Mental Component Score</td>
                                            <td align="center">1.8 ± 5.9</td>
                                            <td align="center">6.2 ± 7.1</td>
                                            <td align="center">4.4 (2.3 to 6.5)</td>
                                            <td align="center">&lt;0.001</td>
                                        </tr>
                                    </tbody>
                                </table>
                                <table-wrap-foot>
                                    <fn>
                                        <p>Data presented as mean ± standard deviation for change from baseline. Between-group differences calculated using analysis of covariance (ANCOVA) with baseline values as covariates. CI = confidence interval; LDL = low-density lipoprotein; HDL = high-density lipoprotein; SF-36 = Short Form 36 Health Survey.</p>
                                    </fn>
                                </table-wrap-foot>
                            </table-wrap>
                        </sec>
                        
                        <sec id="subgroup-analysis">
                            <title>Subgroup Analyses</title>
                            <p>Subgroup analyses were performed to evaluate treatment effects across different patient populations. Consistent benefits were observed across all predefined subgroups, with no evidence of heterogeneity of treatment effect.</p>
                            
                            <fig id="fig4" position="float">
                                <label>Figure 4</label>
                                <caption>
                                    <title>Subgroup Analysis Forest Plot</title>
                                    <p>Forest plot showing treatment effects across predefined subgroups with point estimates and 95% confidence intervals. Subgroups include age categories (&lt;65 vs ≥65 years), gender, diabetes status, BMI categories, and baseline risk score levels. The vertical line represents no treatment effect, points to the left favor intervention, and points to the right favor control. No significant interactions were observed (all p-interaction &gt;0.05).</p>
                                </caption>
                                <graphic xlink:href="figure4_subgroup.tiff" mimetype="image" mime-subtype="tiff"/>
                                <alternatives>
                                    <graphic xlink:href="figure4_subgroup.png" mimetype="image" mime-subtype="png"/>
                                </alternatives>
                            </fig>
                            
                            <table-wrap id="tab3" position="float">
                                <label>Table 3</label>
                                <caption>
                                    <title>Subgroup Analysis of Primary Efficacy Endpoint</title>
                                    <p>Analysis of primary efficacy endpoint (cardiovascular risk score change) across prespecified patient subgroups. Treatment effects shown as mean change from baseline with between-group differences and interaction p-values. Consistent benefits observed across all subgroups with no significant interactions.</p>
                                </caption>
                                <table frame="hsides" rules="groups">
                                    <thead>
                                        <tr>
                                            <th align="left">Subgroup</th>
                                            <th align="center">Control Group<break/>Mean Change</th>
                                            <th align="center">Intervention Group<break/>Mean Change</th>
                                            <th align="center">Treatment Effect<break/>(95% CI)</th>
                                            <th align="center">P-interaction</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td><bold>Overall Population</bold></td>
                                            <td align="center">-4.2 ± 5.9</td>
                                            <td align="center">-12.4 ± 6.8</td>
                                            <td align="center">-8.2 (-10.4 to -6.0)</td>
                                            <td align="center">—</td>
                                        </tr>
                                        <tr>
                                            <td colspan="5"><bold>Age Group</bold></td>
                                        </tr>
                                        <tr>
                                            <td>&lt;65 years (n=89)</td>
                                            <td align="center">-4.8 ± 6.2</td>
                                            <td align="center">-13.1 ± 7.1</td>
                                            <td align="center">-8.3 (-11.2 to -5.4)</td>
                                            <td align="center" rowspan="2">0.892</td>
                                        </tr>
                                        <tr>
                                            <td>≥65 years (n=61)</td>
                                            <td align="center">-3.4 ± 5.4</td>
                                            <td align="center">-11.2 ± 6.2</td>
                                            <td align="center">-7.8 (-11.4 to -4.2)</td>
                                        </tr>
                                        <tr>
                                            <td colspan="5"><bold>Gender</bold></td>
                                        </tr>
                                        <tr>
                                            <td>Male (n=88)</td>
                                            <td align="center">-4.6 ± 6.1</td>
                                            <td align="center">-12.8 ± 7.3</td>
                                            <td align="center">-8.2 (-11.1 to -5.3)</td>
                                            <td align="center" rowspan="2">0.978</td>
                                        </tr>
                                        <tr>
                                            <td>Female (n=62)</td>
                                            <td align="center">-3.7 ± 5.6</td>
                                            <td align="center">-11.8 ± 5.9</td>
                                            <td align="center">-8.1 (-11.8 to -4.4)</td>
                                        </tr>
                                        <tr>
                                            <td colspan="5"><bold>Diabetes Status</bold></td>
                                        </tr>
                                        <tr>
                                            <td>Diabetic (n=60)</td>
                                            <td align="center">-3.8 ± 5.2</td>
                                            <td align="center">-11.9 ± 6.4</td>
                                            <td align="center">-8.1 (-11.3 to -4.9)</td>
                                            <td align="center" rowspan="2">0.876</td>
                                        </tr>
                                        <tr>
                                            <td>Non-diabetic (n=90)</td>
                                            <td align="center">-4.5 ± 6.3</td>
                                            <td align="center">-12.7 ± 7.1</td>
                                            <td align="center">-8.2 (-11.7 to -4.7)</td>
                                        </tr>
                                    </tbody>
                                </table>
                                <table-wrap-foot>
                                    <fn>
                                        <p>Data presented as mean ± standard deviation for change from baseline. Treatment effects calculated as between-group differences using ANCOVA with baseline values as covariates. P-interaction values test for heterogeneity of treatment effect across subgroup categories. CI = confidence interval.</p>
                                    </fn>
                                </table-wrap-foot>
                            </table-wrap>
                        </sec>
                        
                        <sec id="safety-analysis">
                            <title>Safety and Tolerability</title>
                            <p>The intervention was generally well-tolerated with an acceptable safety profile. No new safety signals were identified, and the overall adverse event rate was similar between groups. Most adverse events were mild to moderate in severity and resolved without intervention.</p>
                            
                            <table-wrap id="tab4" position="float">
                                <label>Table 4</label>
                                <caption>
                                    <title>Summary of Adverse Events and Safety Parameters</title>
                                    <p>Comprehensive safety analysis showing adverse event rates, serious adverse events, and laboratory abnormalities during the 24-week study period. Risk ratios calculated for between-group comparisons with 95% confidence intervals. Most adverse events were mild to moderate in severity.</p>
                                </caption>
                                <table frame="hsides" rules="groups">
                                    <thead>
                                        <tr>
                                            <th align="left">Safety Parameter</th>
                                            <th align="center">Control Group<break/>n (%)</th>
                                            <th align="center">Intervention Group<break/>n (%)</th>
                                            <th align="center">Risk Ratio<break/>(95% CI)</th>
                                            <th align="center">P-value</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td colspan="5"><bold>Overall Safety Summary</bold></td>
                                        </tr>
                                        <tr>
                                            <td>Any adverse event</td>
                                            <td align="center">52 (69.3)</td>
                                            <td align="center">48 (64.0)</td>
                                            <td align="center">0.92 (0.73-1.16)</td>
                                            <td align="center">0.489</td>
                                        </tr>
                                        <tr>
                                            <td>Serious adverse events</td>
                                            <td align="center">8 (10.7)</td>
                                            <td align="center">6 (8.0)</td>
                                            <td align="center">0.75 (0.28-2.01)</td>
                                            <td align="center">0.571</td>
                                        </tr>
                                        <tr>
                                            <td>Adverse events leading to discontinuation</td>
                                            <td align="center">3 (4.0)</td>
                                            <td align="center">2 (2.7)</td>
                                            <td align="center">0.67 (0.11-3.93)</td>
                                            <td align="center">0.653</td>
                                        </tr>
                                        <tr>
                                            <td colspan="5"><bold>Common Adverse Events (≥5% in either group)</bold></td>
                                        </tr>
                                        <tr>
                                            <td>Headache</td>
                                            <td align="center">18 (24.0)</td>
                                            <td align="center">15 (20.0)</td>
                                            <td align="center">0.83 (0.46-1.50)</td>
                                            <td align="center">0.542</td>
                                        </tr>
                                        <tr>
                                            <td>Nausea</td>
                                            <td align="center">12 (16.0)</td>
                                            <td align="center">14 (18.7)</td>
                                            <td align="center">1.17 (0.58-2.35)</td>
                                            <td align="center">0.668</td>
                                        </tr>
                                        <tr>
                                            <td>Dizziness</td>
                                            <td align="center">11 (14.7)</td>
                                            <td align="center">8 (10.7)</td>
                                            <td align="center">0.73 (0.32-1.66)</td>
                                            <td align="center">0.448</td>
                                        </tr>
                                        <tr>
                                            <td>Fatigue</td>
                                            <td align="center">9 (12.0)</td>
                                            <td align="center">11 (14.7)</td>
                                            <td align="center">1.22 (0.54-2.78)</td>
                                            <td align="center">0.634</td>
                                        </tr>
                                    </tbody>
                                </table>
                                <table-wrap-foot>
                                    <fn>
                                        <p>Adverse events reported by ≥5% of participants in either treatment group. Risk ratios and 95% confidence intervals calculated using exact methods. P-values from Fisher's exact test for categorical comparisons. CI = confidence interval.</p>
                                    </fn>
                                </table-wrap-foot>
                            </table-wrap>
                        </sec>
                    </sec>
                    
                    <sec id="discussion">
                        <title>Discussion</title>
                        <p>This randomized controlled trial demonstrates that the novel therapeutic intervention provides significant clinical benefits compared to standard care in patients with cardiovascular disease. The primary efficacy endpoint showed a clinically meaningful improvement with strong statistical significance.</p>
                        <p>The magnitude and consistency of treatment effects observed in this study exceed those reported in previous trials of similar interventions. As shown in <xref ref-type="fig" rid="fig3">Figure 3</xref>, the intervention group demonstrated superior outcomes across all quartiles of response, indicating broad therapeutic benefit.</p>
                        <p>The safety profile was acceptable and comparable to control, with no unexpected adverse events or safety signals identified as detailed in <xref ref-type="table" rid="tab4">Table 4</xref>.</p>
                    </sec>
                    
                    <sec id="conclusions">
                        <title>Conclusions</title>
                        <p>This randomized controlled trial provides robust evidence for the efficacy and safety of the novel therapeutic intervention in patients with cardiovascular disease. The intervention demonstrated statistically significant and clinically meaningful improvements in the primary efficacy endpoint and all secondary endpoints.</p>
                        <p>The consistent treatment effects observed across diverse patient subgroups and the magnitude of clinical benefits suggest this intervention may address an important unmet medical need in cardiovascular medicine.</p>
                    </sec>
                </body>
            </article>""",
            "extraction_method": "lxml",
            "schema_type": "pmc",
            "namespaces": {
                "xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "mml": "http://www.w3.org/1998/Math/MathML",
                "xlink": "http://www.w3.org/1999/xlink"
            }
        }
    
    @staticmethod
    def get_edge_case_pdf_content() -> Dict[str, Any]:
        """Get edge case PDF content for testing error handling."""
        return {
            "text": """
            Edge Case Document with Various Figure and Table Formats
            
            Figure 1.2a. Subfigure with complex numbering.
            Fig. A-1: Appendix figure with dash notation.
            Figure S1. Supplementary material figure.
            FIGURE II. Roman numeral figure.
            
            Table 1.1. Hierarchical table numbering.
            Tab. B-2: Appendix table format.
            Table S2. Supplementary table.
            TABLE III. Roman numeral table.
            
            Figure with missing caption
            
            Table without proper caption
            
            Figure 999. Figure with very long caption that goes on and on describing the experimental setup in excruciating detail including every possible parameter, measurement technique, statistical analysis method, quality control procedure, and validation step that was performed during the course of this comprehensive and thorough scientific investigation.
            
            Table 999. Table with missing data cells and incomplete information.
            Column 1 | Column 2 | Column 3
            Data     |          | More
                     | Missing  |
            
            Fig. 1α. Figure with Unicode characters in label.
            Tableau 1. Figure with non-English label.
            
            Figure 1 (continued). Continuation figure.
            Table 1 - Part A. Multi-part table.
            """,
            "extraction_method": "test_edge_cases",
            "pages": 3,
            "raw_content": None
        }
    
    @staticmethod
    def get_edge_case_xml_content() -> Dict[str, Any]:
        """Get edge case XML content for testing error handling."""
        return {
            "xml_content": """<?xml version="1.0" encoding="UTF-8"?>
            <article>
                <!-- Figure with minimal information -->
                <fig id="minimal-fig">
                    <caption><p></p></caption>
                </fig>
                
                <!-- Figure without ID -->
                <fig>
                    <label>Unlabeled Figure</label>
                    <caption><p>Figure without proper ID attribute</p></caption>
                </fig>
                
                <!-- Complex figure with multiple graphics -->
                <fig id="complex-fig" position="float">
                    <label>Figure 1</label>
                    <caption>
                        <title>Complex Multi-part Figure</title>
                        <p>This figure consists of multiple parts: (A) experimental setup, (B) control conditions, (C) results visualization, and (D) statistical analysis. Each component provides different perspectives on the same experimental data.</p>
                    </caption>
                    <graphic href="fig1a.tiff" mimetype="image" mime-subtype="tiff"/>
                    <graphic href="fig1b.png" mimetype="image" mime-subtype="png"/>
                    <alternatives>
                        <graphic href="fig1_alt.eps" mimetype="application" mime-subtype="postscript"/>
                    </alternatives>
                </fig>
                
                <!-- Table with minimal structure -->
                <table-wrap id="minimal-table">
                    <caption><p>Minimal table</p></caption>
                    <table>
                        <tr><td>Data</td></tr>
                    </table>
                </table-wrap>
                
                <!-- Table without ID -->
                <table-wrap>
                    <label>Unlabeled Table</label>
                    <caption><p>Table without proper ID</p></caption>
                    <table>
                        <tr><th>Header</th></tr>
                        <tr><td>Data</td></tr>
                    </table>
                </table-wrap>
                
                <!-- Complex table with merged cells -->
                <table-wrap id="complex-table" position="float">
                    <label>Table 1</label>
                    <caption>
                        <title>Complex Table with Merged Cells</title>
                        <p>Comprehensive data table showing experimental results with hierarchical headers, merged cells, and statistical annotations.</p>
                    </caption>
                    <table frame="hsides" rules="groups">
                        <thead>
                            <tr>
                                <th rowspan="2">Parameter</th>
                                <th colspan="3">Treatment Groups</th>
                                <th rowspan="2">P-value</th>
                            </tr>
                            <tr>
                                <th>Control</th>
                                <th>Low Dose</th>
                                <th>High Dose</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Measurement 1</td>
                                <td>12.5 ± 2.1</td>
                                <td>15.7 ± 2.8</td>
                                <td>18.2 ± 3.1</td>
                                <td>&lt;0.001</td>
                            </tr>
                            <tr>
                                <td>Measurement 2</td>
                                <td colspan="3">Not measured</td>
                                <td>—</td>
                            </tr>
                        </tbody>
                        <tfoot>
                            <tr>
                                <td colspan="5">Data presented as mean ± SD</td>
                            </tr>
                        </tfoot>
                    </table>
                    <table-wrap-foot>
                        <fn>
                            <p>Additional notes and statistical information.</p>
                        </fn>
                    </table-wrap-foot>
                </table-wrap>
                
                <!-- Unicode content -->
                <fig id="unicode-fig">
                    <caption>
                        <p>Figure with Unicode: α=0.05, β-test, ±2.5°C, µg/mL, ≥99% purity</p>
                    </caption>
                </fig>
                
                <!-- Nested structures -->
                <fig-group id="figure-group">
                    <label>Figure Group 1</label>
                    <caption><p>Group of related figures</p></caption>
                    <fig id="subfig-a">
                        <label>A</label>
                        <caption><p>Subfigure A</p></caption>
                    </fig>
                    <fig id="subfig-b">
                        <label>B</label>
                        <caption><p>Subfigure B</p></caption>
                    </fig>
                </fig-group>
            </article>""",
            "extraction_method": "etree",
            "schema_type": "edge_cases"
        }
    
    @staticmethod
    def get_performance_test_content() -> Dict[str, Any]:
        """Get large content for performance testing."""
        # Generate large document content
        large_text = "Introduction to performance testing. " * 500
        
        # Add many figures
        for i in range(100):
            large_text += f"\nFigure {i+1}. Performance test figure {i+1} with detailed caption describing experimental methodology and statistical analysis. "
            large_text += "Additional descriptive text. " * 20
        
        # Add many tables
        for i in range(100):
            large_text += f"\nTable {i+1}. Performance test table {i+1} with comprehensive data analysis. "
            large_text += "Column 1 | Column 2 | Column 3\n"
            large_text += "Data A   | Data B   | Data C\n" * 10
        
        return {
            "text": large_text,
            "extraction_method": "performance_test",
            "pages": 200,
            "raw_content": Mock()
        }


@pytest.fixture
def sample_pdf_content():
    """Fixture providing sample PDF content."""
    return MockTestData.get_sample_pdf_content()


@pytest.fixture
def sample_xml_content():
    """Fixture providing sample XML content."""
    return MockTestData.get_sample_xml_content()


@pytest.fixture
def edge_case_pdf_content():
    """Fixture providing edge case PDF content."""
    return MockTestData.get_edge_case_pdf_content()


@pytest.fixture
def edge_case_xml_content():
    """Fixture providing edge case XML content."""
    return MockTestData.get_edge_case_xml_content()


@pytest.fixture
def performance_test_content():
    """Fixture providing large content for performance testing."""
    return MockTestData.get_performance_test_content()


@pytest.fixture
def mock_pdf_libraries():
    """Fixture for mocking PDF library availability."""
    with patch('aim2_project.aim2_ontology.parsers.pdf_parser.PYPDF_AVAILABLE', True), \
         patch('aim2_project.aim2_ontology.parsers.pdf_parser.PDFPLUMBER_AVAILABLE', True), \
         patch('aim2_project.aim2_ontology.parsers.pdf_parser.FITZ_AVAILABLE', True):
        yield


@pytest.fixture
def mock_xml_libraries():
    """Fixture for mocking XML library availability."""
    with patch('aim2_project.aim2_ontology.parsers.xml_parser.LXML_AVAILABLE', True):
        yield


@pytest.fixture
def content_extractor():
    """Fixture providing ContentExtractor instance."""
    return ContentExtractor()


@pytest.fixture
def quality_assessor():
    """Fixture providing QualityAssessment instance."""
    return QualityAssessment()


# Test data validation functions
def validate_figure_metadata(figure_dict: Dict[str, Any]) -> bool:
    """Validate figure metadata structure."""
    required_keys = {"id", "type", "caption"}
    return all(key in figure_dict for key in required_keys)


def validate_table_metadata(table_dict: Dict[str, Any]) -> bool:
    """Validate table metadata structure."""
    required_keys = {"id", "type", "caption"}
    return all(key in table_dict for key in required_keys)


def validate_quality_metrics(quality_dict: Dict[str, Any]) -> bool:
    """Validate quality metrics structure."""
    required_keys = {"extraction_confidence", "overall_quality"}
    return all(key in quality_dict for key in required_keys)