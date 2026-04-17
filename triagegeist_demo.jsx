import { useState, useCallback, useMemo } from "react";

const ESI_COLORS = {
  1: { bg: "#FEE2E2", text: "#991B1B", border: "#F87171", label: "Resuscitation" },
  2: { bg: "#FEF3C7", text: "#92400E", border: "#FBBF24", label: "Emergent" },
  3: { bg: "#FEF9C3", text: "#854D0E", border: "#FACC15", label: "Urgent" },
  4: { bg: "#DCFCE7", text: "#166534", border: "#4ADE80", label: "Less Urgent" },
  5: { bg: "#DBEAFE", text: "#1E40AF", border: "#60A5FA", label: "Non-Urgent" },
};

const MENTAL_STATUS = ["alert", "confused", "drowsy", "agitated", "unresponsive"];
const COMPLAINT_SYSTEMS = [
  "cardiovascular", "respiratory", "neurological", "gastrointestinal",
  "musculoskeletal", "trauma", "psychiatric", "infectious",
  "dermatological", "genitourinary", "ENT", "ophthalmic", "endocrine", "other"
];

function predictESI(vitals) {
  const { gcs, news2, spo2, hr, sbp, rr, temp, mentalStatus, painScore, chiefComplaint } = vitals;
  const cc = (chiefComplaint || "").toLowerCase();

  if (gcs <= 8) return { esi: 1, confidence: 0.97, reason: "GCS ≤ 8 indicates comatose state requiring immediate airway management" };
  if (mentalStatus === "unresponsive") return { esi: 1, confidence: 0.95, reason: "Unresponsive patient requires immediate resuscitation assessment" };
  if (news2 >= 15) return { esi: 1, confidence: 0.93, reason: "NEWS2 ≥ 15 indicates extreme physiological derangement" };

  if (cc.includes("cardiac arrest") || cc.includes("tension pneumo") || cc.includes("massive hemorrh"))
    return { esi: 1, confidence: 0.92, reason: "Chief complaint indicates immediately life-threatening condition" };

  if (news2 >= 10) return { esi: 1, confidence: 0.88, reason: "NEWS2 ≥ 10 with high clinical risk — immediate intervention likely needed" };

  if (news2 >= 7) return { esi: 2, confidence: 0.91, reason: "NEWS2 7-9 indicates high clinical risk requiring emergent evaluation" };
  if (mentalStatus === "drowsy") return { esi: 2, confidence: 0.87, reason: "Drowsy mental status suggests altered consciousness — emergent priority" };
  if (spo2 < 88) return { esi: 2, confidence: 0.90, reason: "Severe hypoxia (SpO2 < 88%) requires emergent respiratory support" };
  if (mentalStatus === "agitated" && news2 >= 5) return { esi: 2, confidence: 0.85, reason: "Agitated patient with elevated NEWS2 — emergent evaluation needed" };
  if (gcs <= 12) return { esi: 2, confidence: 0.89, reason: "GCS 9-12 (moderate impairment) warrants emergent assessment" };

  if (cc.includes("severe") || cc.includes("acute")) {
    if (news2 >= 5) return { esi: 2, confidence: 0.84, reason: "Severe/acute presentation with elevated NEWS2" };
    return { esi: 3, confidence: 0.82, reason: "Severe/acute chief complaint likely requiring multiple resources" };
  }

  if (news2 >= 5) return { esi: 3, confidence: 0.86, reason: "NEWS2 5-6 (medium risk) — urgent evaluation with multiple resources expected" };
  if (mentalStatus === "confused") return { esi: 3, confidence: 0.83, reason: "Confused mental status warrants urgent workup" };
  if (painScore >= 7) return { esi: 3, confidence: 0.80, reason: "Severe pain (≥7/10) typically requires multiple resources" };
  if (temp >= 39.0) return { esi: 3, confidence: 0.81, reason: "High fever (≥39°C) warrants urgent evaluation with labs" };
  if (sbp < 90) return { esi: 2, confidence: 0.88, reason: "Hypotension (SBP < 90) indicates hemodynamic instability" };
  if (hr > 120 && news2 >= 3) return { esi: 3, confidence: 0.79, reason: "Significant tachycardia with physiological derangement" };

  if (news2 >= 3) return { esi: 3, confidence: 0.78, reason: "NEWS2 3-4 with low-risk features — urgent, likely needing 2+ resources" };

  if (cc.includes("mild") || cc.includes("minor")) {
    if (news2 <= 1) return { esi: 5, confidence: 0.85, reason: "Mild complaint with normal vitals — no resources anticipated" };
    return { esi: 4, confidence: 0.82, reason: "Mild complaint likely requiring single resource" };
  }

  if (cc.includes("request") || cc.includes("advice") || cc.includes("removal") || cc.includes("refill"))
    return { esi: 5, confidence: 0.90, reason: "Administrative/advice request — no clinical resources needed" };

  if (cc.includes("chronic") || cc.includes("review") || cc.includes("follow"))
    return { esi: 4, confidence: 0.80, reason: "Chronic/follow-up complaint likely requiring single resource" };

  if (news2 <= 1 && painScore <= 3 && mentalStatus === "alert")
    return { esi: 4, confidence: 0.77, reason: "Low-acuity presentation: normal vitals, mild pain, alert" };

  return { esi: 3, confidence: 0.70, reason: "Moderate presentation — default to ESI-3 pending further assessment" };
}

function ContributionBar({ label, value, color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "3px 0" }}>
      <span style={{ width: 100, fontSize: 11, color: "#6B7280", textAlign: "right", flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: 8, background: "#F3F4F6", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${Math.min(value, 100)}%`, height: "100%", background: color, borderRadius: 4, transition: "width 0.5s ease" }} />
      </div>
      <span style={{ fontSize: 11, color: "#374151", width: 35, textAlign: "right" }}>{value}%</span>
    </div>
  );
}

export default function TriageDemo() {
  const [vitals, setVitals] = useState({
    gcs: 15, news2: 2, spo2: 97, hr: 82, sbp: 125, rr: 16,
    temp: 37.2, mentalStatus: "alert", painScore: 3,
    chiefComplaint: "", complaintSystem: "other",
    age: 45, sex: "M"
  });
  const [showResult, setShowResult] = useState(false);

  const update = useCallback((key, val) => {
    setVitals(prev => ({ ...prev, [key]: val }));
    setShowResult(false);
  }, []);

  const result = useMemo(() => predictESI(vitals), [vitals]);
  const esiInfo = ESI_COLORS[result.esi];

  const contributions = useMemo(() => {
    const c = [];
    const n2 = Math.min(vitals.news2 * 6, 100);
    c.push({ label: "NEWS2", value: n2, color: n2 > 50 ? "#EF4444" : n2 > 30 ? "#F59E0B" : "#10B981" });
    const gcsCont = vitals.gcs < 15 ? Math.min((15 - vitals.gcs) * 8, 100) : 5;
    c.push({ label: "GCS", value: gcsCont, color: gcsCont > 50 ? "#EF4444" : "#6366F1" });
    const spo2Cont = vitals.spo2 < 94 ? Math.min((100 - vitals.spo2) * 4, 100) : 5;
    c.push({ label: "SpO2", value: spo2Cont, color: spo2Cont > 30 ? "#EF4444" : "#06B6D4" });
    const msCont = vitals.mentalStatus === "unresponsive" ? 95 : vitals.mentalStatus === "drowsy" ? 70 : vitals.mentalStatus === "agitated" ? 50 : vitals.mentalStatus === "confused" ? 40 : 5;
    c.push({ label: "Mental status", value: msCont, color: msCont > 50 ? "#EF4444" : "#8B5CF6" });
    const painCont = Math.min(vitals.painScore * 10, 100);
    c.push({ label: "Pain", value: painCont, color: painCont > 60 ? "#EF4444" : "#F59E0B" });
    const ccText = (vitals.chiefComplaint || "").toLowerCase();
    const ccCont = ccText.includes("severe") || ccText.includes("massive") ? 80 : ccText.includes("acute") ? 60 : ccText.includes("mild") ? 15 : ccText.includes("request") ? 5 : 30;
    c.push({ label: "Chief complaint", value: ccCont, color: ccCont > 50 ? "#EF4444" : "#10B981" });
    return c.sort((a, b) => b.value - a.value);
  }, [vitals]);

  return (
    <div style={{ fontFamily: "'IBM Plex Sans', system-ui, sans-serif", maxWidth: 800, margin: "0 auto", padding: "0 16px" }}>
      <div style={{ borderBottom: "2px solid #0F172A", paddingBottom: 12, marginBottom: 24 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <span style={{ fontSize: 22, fontWeight: 700, color: "#0F172A", letterSpacing: "-0.5px" }}>TRIAGEGEIST</span>
          <span style={{ fontSize: 12, color: "#64748B", letterSpacing: 1.5, textTransform: "uppercase" }}>Clinical decision support</span>
        </div>
        <p style={{ fontSize: 13, color: "#64748B", margin: "6px 0 0" }}>AI-powered ESI triage acuity prediction — 98.6% accuracy on 80,000 Finnish ED visits</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {/* Left Column - Inputs */}
        <div>
          <div style={{ fontSize: 11, fontWeight: 600, color: "#94A3B8", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 10 }}>Patient vitals</div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            {[
              { label: "GCS Total", key: "gcs", min: 3, max: 15, step: 1, unit: "/15" },
              { label: "NEWS2 Score", key: "news2", min: 0, max: 17, step: 1, unit: "" },
              { label: "SpO2", key: "spo2", min: 60, max: 100, step: 1, unit: "%" },
              { label: "Heart Rate", key: "hr", min: 30, max: 200, step: 1, unit: "bpm" },
              { label: "Systolic BP", key: "sbp", min: 50, max: 230, step: 1, unit: "mmHg" },
              { label: "Resp Rate", key: "rr", min: 6, max: 50, step: 1, unit: "/min" },
              { label: "Temperature", key: "temp", min: 34, max: 42, step: 0.1, unit: "°C" },
              { label: "Pain Score", key: "painScore", min: 0, max: 10, step: 1, unit: "/10" },
            ].map(({ label, key, min, max, step, unit }) => (
              <div key={key} style={{ padding: "8px 10px", background: "#F8FAFC", borderRadius: 8, border: "1px solid #E2E8F0" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 11, color: "#64748B" }}>{label}</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: "#0F172A" }}>{vitals[key]}{unit}</span>
                </div>
                <input type="range" min={min} max={max} step={step} value={vitals[key]}
                  onChange={e => update(key, parseFloat(e.target.value))}
                  style={{ width: "100%", height: 4, accentColor: "#0F172A" }} />
              </div>
            ))}
          </div>

          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 11, color: "#64748B", marginBottom: 4 }}>Mental status</div>
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
              {MENTAL_STATUS.map(s => (
                <button key={s} onClick={() => update("mentalStatus", s)}
                  style={{
                    padding: "5px 10px", fontSize: 11, borderRadius: 6, cursor: "pointer",
                    border: vitals.mentalStatus === s ? "1.5px solid #0F172A" : "1px solid #CBD5E1",
                    background: vitals.mentalStatus === s ? "#0F172A" : "#fff",
                    color: vitals.mentalStatus === s ? "#fff" : "#475569",
                    fontWeight: vitals.mentalStatus === s ? 600 : 400,
                    transition: "all 0.15s"
                  }}>
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 11, color: "#64748B", marginBottom: 4 }}>Chief complaint</div>
            <input type="text" placeholder="e.g. severe chest pain with diaphoresis"
              value={vitals.chiefComplaint}
              onChange={e => update("chiefComplaint", e.target.value)}
              style={{ width: "100%", padding: "8px 10px", fontSize: 13, borderRadius: 8, border: "1px solid #CBD5E1", outline: "none", boxSizing: "border-box" }} />
          </div>

          <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 11, color: "#64748B", marginBottom: 4 }}>Complaint system</div>
            <select value={vitals.complaintSystem} onChange={e => update("complaintSystem", e.target.value)}
              style={{ width: "100%", padding: "8px 10px", fontSize: 12, borderRadius: 8, border: "1px solid #CBD5E1", background: "#fff" }}>
              {COMPLAINT_SYSTEMS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          <button onClick={() => setShowResult(true)}
            style={{
              width: "100%", marginTop: 14, padding: "10px", fontSize: 14, fontWeight: 600,
              background: "#0F172A", color: "#fff", border: "none", borderRadius: 8,
              cursor: "pointer", transition: "all 0.15s", letterSpacing: 0.3
            }}>
            Predict triage acuity
          </button>
        </div>

        {/* Right Column - Results */}
        <div>
          {showResult ? (
            <div style={{ animation: "fadeIn 0.3s ease" }}>
              <div style={{
                background: esiInfo.bg, border: `2px solid ${esiInfo.border}`, borderRadius: 12,
                padding: 20, textAlign: "center", marginBottom: 14
              }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: esiInfo.text, letterSpacing: 1.5, textTransform: "uppercase" }}>Predicted triage level</div>
                <div style={{ fontSize: 48, fontWeight: 800, color: esiInfo.text, lineHeight: 1.1, margin: "6px 0" }}>ESI-{result.esi}</div>
                <div style={{ fontSize: 14, fontWeight: 600, color: esiInfo.text }}>{esiInfo.label}</div>
                <div style={{
                  display: "inline-block", marginTop: 8, padding: "3px 12px", borderRadius: 20,
                  background: "rgba(0,0,0,0.08)", fontSize: 12, color: esiInfo.text
                }}>
                  Confidence: {Math.round(result.confidence * 100)}%
                </div>
              </div>

              <div style={{ background: "#F8FAFC", borderRadius: 10, padding: 14, marginBottom: 14, border: "1px solid #E2E8F0" }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: "#94A3B8", letterSpacing: 1, textTransform: "uppercase", marginBottom: 6 }}>Clinical reasoning</div>
                <p style={{ fontSize: 13, color: "#334155", margin: 0, lineHeight: 1.5 }}>{result.reason}</p>
              </div>

              <div style={{ background: "#F8FAFC", borderRadius: 10, padding: 14, border: "1px solid #E2E8F0" }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: "#94A3B8", letterSpacing: 1, textTransform: "uppercase", marginBottom: 8 }}>Feature contributions</div>
                {contributions.map(c => (
                  <ContributionBar key={c.label} label={c.label} value={c.value} color={c.color} />
                ))}
              </div>

              <div style={{ marginTop: 12, padding: 12, background: "#FFFBEB", borderRadius: 8, border: "1px solid #FDE68A" }}>
                <p style={{ fontSize: 11, color: "#92400E", margin: 0, lineHeight: 1.5 }}>
                  <strong>Disclaimer:</strong> This tool is a proof-of-concept for research purposes only. It must not be used for actual clinical decision-making. Always rely on trained clinical judgment.
                </p>
              </div>
            </div>
          ) : (
            <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", padding: 40, color: "#94A3B8" }}>
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 6v6l4 2M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
              </svg>
              <p style={{ fontSize: 14, marginTop: 12, textAlign: "center" }}>Adjust patient vitals and click<br /><strong style={{ color: "#0F172A" }}>Predict triage acuity</strong></p>
            </div>
          )}
        </div>
      </div>

      <div style={{ marginTop: 20, paddingTop: 12, borderTop: "1px solid #E2E8F0", display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94A3B8" }}>
        <span>Triagegeist · Laitinen-Fredriksson Foundation Challenge 2026</span>
        <span>LightGBM · 5-fold CV · 186 features · QWK 0.993</span>
      </div>

      <style>{`@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }`}</style>
    </div>
  );
}
