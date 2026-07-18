import { useRef, useState } from "react";
import { LEAF_SAMPLES, type LeafSample } from "../data";
import { api } from "../lib/api";
import { Card, RadialPct, Reveal, SectionHead, StatusChip } from "../lib/ui";

type Phase = "idle" | "scanning" | "done";

export function Vision() {
  const [sample, setSample] = useState<LeafSample>(LEAF_SAMPLES[0]);
  const [uploadUrl, setUploadUrl] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [apiNote, setApiNote] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const applyApiResult = (r: any, base: LeafSample) => {
    if (!r) return base;
    return {
      ...base,
      disease: r.disease || r.prediction || r.label || base.disease,
      conf: Number(r.confidence ?? r.conf ?? base.conf),
      severity: Number(r.severity_pct ?? r.severity ?? base.severity),
      symptomsEn: r.symptoms || r.advice || base.symptomsEn,
      chemEn: r.treatment || r.chemical || base.chemEn,
    } as LeafSample;
  };

  const diagnose = async (s?: LeafSample) => {
    if (s) setSample(s);
    setUploadUrl(null);
    setPhase("scanning");
    setApiNote(null);
    const base = s || sample;
    try {
      const r = await api.visionDiagnose(undefined, base.name);
      setSample(applyApiResult(r, base));
      setApiNote("Live vision API");
    } catch {
      setApiNote("Demo classifier (API offline)");
    }
    setPhase("done");
  };

  const onUpload = async (f: File | undefined) => {
    if (!f) return;
    setUploadUrl(URL.createObjectURL(f));
    setPhase("scanning");
    setApiNote(null);
    try {
      const r = await api.visionDiagnose(f, sample.name);
      setSample(applyApiResult(r, sample));
      setApiNote("Live vision API · uploaded leaf");
    } catch {
      setSample(LEAF_SAMPLES[0]);
      setApiNote("Demo result (upload API offline)");
    }
    setPhase("done");
  };

  const res = sample;
  const sevColor = res.severity > 55 ? "var(--chili)" : res.severity > 25 ? "var(--marigold)" : "var(--leaf)";

  return (
    <div>
      <SectionHead
        eyebrow="Computer vision · पानाचे निदान"
        title={<>Disease <span className="text-[var(--marigold)]">Diagnostic</span> <span className="devnagari text-[18px] font-semibold text-[var(--mut)] ml-1">· रोग तपासणी</span></>}
        sub="Upload a leaf photo or pick a field sample. The CNN classifier returns the disease, confidence, severity and a bilingual treatment protocol."
        right={
          <div className="flex gap-1.5">
            <StatusChip tone="marigold">vision-cnn v2.3</StatusChip>
            {apiNote && <StatusChip tone="leaf">{apiNote}</StatusChip>}
          </div>
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-4">
        {/* left: picker */}
        <div className="space-y-4">
          <Reveal>
            <Card ticks className="p-4">
              <span className="tick-b" />
              <div className="eyebrow mb-3">Field samples</div>
              <div className="grid grid-cols-2 gap-2.5">
                {LEAF_SAMPLES.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => diagnose(s)}
                    className={`group relative rounded-[10px] overflow-hidden border text-left transition-all duration-200 ${
                      sample.id === s.id && !uploadUrl ? "border-[var(--gold)] shadow-[0_0_0_1px_var(--gold),0_8px_20px_-10px_rgba(244,192,75,0.5)]" : "border-[var(--line2)] hover:border-[var(--leaf)]"
                    }`}
                  >
                    <img src={s.img} alt={s.name} className="w-full h-[86px] object-cover transition-transform duration-500 group-hover:scale-105" />
                    <div className="absolute inset-0 bg-gradient-to-t from-[#08130cd9] to-transparent" />
                    <div className="absolute bottom-1.5 left-2 right-2">
                      <div className="text-[11.5px] font-semibold leading-tight">{s.name}</div>
                      <div className="text-[10px] devnagari text-[var(--mut)]">{s.nameMr}</div>
                    </div>
                  </button>
                ))}
              </div>
            </Card>
          </Reveal>

          <Reveal delay={90}>
            <Card className="p-4">
              <div className="eyebrow mb-3">Upload leaf photo</div>
              <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(e) => onUpload(e.target.files?.[0])} />
              <button
                onClick={() => fileRef.current?.click()}
                className="w-full rounded-[10px] border-2 border-dashed border-[var(--line2)] hover:border-[var(--gold)] bg-[rgba(154,205,162,0.03)] hover:bg-[rgba(244,192,75,0.05)] transition-all duration-200 py-7 px-4 text-center group"
              >
                <div className="text-[26px] transition-transform duration-300 group-hover:-translate-y-1">📸</div>
                <div className="text-[13px] font-semibold mt-1">Drop or browse — JPG / PNG</div>
                <div className="text-[11px] text-[var(--dim)] mt-0.5">runs on-device · no photo leaves your phone</div>
              </button>
              <div className="mt-3 text-[11px] font-mono2 text-[var(--dim)]">crop_hint: auto-detected from taxonomy aliases</div>
            </Card>
          </Reveal>
        </div>

        {/* right: analysis */}
        <Reveal delay={60}>
          <Card ticks className="p-4 h-full">
            <span className="tick-b" />
            <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
              <div className="eyebrow">Diagnostic panel</div>
              {phase === "done" && !uploadUrl && <span className="font-mono2 text-[11px] text-[var(--dim)]">{res.zone}</span>}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* image */}
              <div className="relative rounded-[12px] overflow-hidden border border-[var(--line2)] aspect-[4/3] bg-[#0a1810]">
                {uploadUrl ? (
                  <img src={uploadUrl} alt="Uploaded leaf" className="absolute inset-0 w-full h-full object-cover" />
                ) : (
                  <img src={sample.img} alt={sample.name} className="absolute inset-0 w-full h-full object-cover" />
                )}
                {phase === "scanning" && (
                  <>
                    <div className="scanline" />
                    <div className="absolute inset-0 bg-[rgba(6,16,10,0.35)]" />
                    <div className="absolute top-2 left-2 chip chip-leaf">classifying…</div>
                  </>
                )}
                {phase === "done" && (
                  <div className="absolute top-2 left-2 chip chip-gold">✓ analysis complete</div>
                )}
                <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-[#08130ce6] to-transparent px-3 py-2">
                  <div className="font-mono2 text-[10px] tracking-[0.16em] text-[var(--dim)]">
                    {uploadUrl ? "UPLOADED SAMPLE" : `${sample.name.toUpperCase()} · ${sample.nameMr}`}
                  </div>
                </div>
              </div>

              {/* verdict */}
              <div className="flex flex-col">
                <div className="flex items-center gap-4">
                  <RadialPct pct={res.conf * 100} color={sevColor} label="CONFIDENCE" />
                  <div className="min-w-0">
                    <div className="text-[10.5px] font-mono2 uppercase tracking-[0.16em] text-[var(--dim)]">Detected</div>
                    <div className="font-display font-bold text-[19px] leading-tight" style={{ color: sevColor }}>{res.disease}</div>
                    <div className="devnagari text-[14px] text-[var(--mut)]">{res.diseaseMr}</div>
                    <div className="text-[11px] text-[var(--dim)] italic mt-0.5">{res.scientific}</div>
                  </div>
                </div>
                <div className="mt-3">
                  <div className="flex justify-between text-[10.5px] font-mono2 uppercase tracking-[0.14em] text-[var(--dim)] mb-1">
                    <span>Severity</span><span style={{ color: sevColor }}>{res.severity}/100</span>
                  </div>
                  <div className="h-[7px] rounded-full bg-[rgba(154,205,162,0.1)] overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-1000" style={{ width: `${res.severity}%`, background: `linear-gradient(90deg, var(--leaf), ${sevColor})` }} />
                  </div>
                </div>
                <div className="mt-4 space-y-1.5">
                  {res.diff.map((d, i) => (
                    <div key={d.name} className="flex items-center gap-2 text-[11.5px]">
                      <span className="w-[38%] truncate text-[var(--mut)]">{d.name}</span>
                      <div className="flex-1 h-[5px] rounded-full bg-[rgba(154,205,162,0.1)] overflow-hidden">
                        <div className="h-full rounded-full bar-grow-x" style={{ width: `${d.pct}%`, background: i === 0 ? sevColor : "var(--dim)", animationDelay: `${i * 90}ms` }} />
                      </div>
                      <span className="font-mono2 text-[var(--dim)] w-[34px] text-right">{d.pct}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {phase === "done" && (
              <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
                {[
                  { icon: "🔍", title: "Symptoms", mr: res.symptomsMr, en: res.symptomsEn, tone: "var(--gold)" },
                  { icon: "🌿", title: "Organic control", mr: res.organicMr, en: res.organicEn, tone: "var(--leaf)" },
                  { icon: "🧪", title: "Chemical control", mr: res.chemMr, en: res.chemEn, tone: "var(--water)" },
                ].map((t, i) => (
                  <div key={t.title} className="rounded-[11px] border border-[var(--line)] bg-[rgba(154,205,162,0.03)] p-3.5 reveal" style={{ animationDelay: `${i * 110}ms` }}>
                    <div className="flex items-center gap-2 mb-2">
                      <span>{t.icon}</span>
                      <span className="font-display font-bold text-[13.5px]" style={{ color: t.tone }}>{t.title}</span>
                    </div>
                    <p className="text-[12.5px] devnagari text-[var(--ink)] leading-relaxed">{t.mr}</p>
                    <p className="text-[11.5px] text-[var(--mut)] leading-relaxed mt-1.5">{t.en}</p>
                  </div>
                ))}
              </div>
            )}

            {phase === "idle" && (
              <div className="mt-6 text-center py-8 text-[var(--dim)]">
                <div className="text-[28px] mb-2">🩺</div>
                <p className="text-[13px]">Select a sample or upload a photo to start the diagnostic.</p>
                <button className="btn btn-leaf mt-3" onClick={() => diagnose()}>Analyze sample leaf</button>
              </div>
            )}
          </Card>
        </Reveal>
      </div>
    </div>
  );
}
