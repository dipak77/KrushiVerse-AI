import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { Card, Reveal, SectionHead, StatusChip } from "../lib/ui";

export function Vision() {
  const [samples, setSamples] = useState<any[]>([]);
  const [sample, setSample] = useState<any>(null);
  const [uploadUrl, setUploadUrl] = useState<string | null>(null);
  const [phase, setPhase] = useState<"idle" | "scanning" | "done">("idle");
  const [note, setNote] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    (async () => {
      try {
        const r = await api.visionSamples();
        setSamples(r.samples || []);
        setSample((r.samples || [])[0] || null);
        setNote("samples from /api/ui/vision/samples");
      } catch (e: any) {
        setErr(e?.message || String(e));
      }
    })();
  }, []);

  const diagnoseSample = async (s: any) => {
    setSample(s);
    setUploadUrl(null);
    setPhase("scanning");
    try {
      const r = await api.visionDiagnose(undefined, s.name);
      setSample({
        ...s,
        disease: r.disease_identified_en || s.disease,
        diseaseMr: r.disease_identified_mr || s.diseaseMr,
        conf: r.confidence_score ?? s.conf,
        symptomsEn: r.symptoms_en || s.symptomsEn,
        symptomsMr: r.symptoms_mr || s.symptomsMr,
        organicEn: r.organic_treatment?.en || s.organicEn,
        organicMr: r.organic_treatment?.mr || s.organicMr,
        chemEn: r.chemical_treatment?.en || s.chemEn,
        chemMr: r.chemical_treatment?.mr || s.chemMr,
      });
      setNote("diagnosis from /api/vision/diagnose");
    } catch (e: any) {
      setNote(`sample catalog only (${e?.message || "diagnose failed"})`);
    }
    setPhase("done");
  };

  const onUpload = async (f?: File) => {
    if (!f) return;
    setUploadUrl(URL.createObjectURL(f));
    setPhase("scanning");
    try {
      const r = await api.visionDiagnose(f, sample?.name || "Pomegranate");
      setSample({
        ...(sample || {}),
        name: r.detected_crop || sample?.name,
        disease: r.disease_identified_en,
        diseaseMr: r.disease_identified_mr,
        conf: r.confidence_score,
        symptomsEn: r.symptoms_en,
        symptomsMr: r.symptoms_mr,
        organicEn: r.organic_treatment?.en,
        organicMr: r.organic_treatment?.mr,
        chemEn: r.chemical_treatment?.en,
        chemMr: r.chemical_treatment?.mr,
        img: uploadUrl,
      });
      setNote("upload diagnosis from /api/vision/diagnose");
    } catch (e: any) {
      setErr(e?.message || String(e));
    }
    setPhase("done");
  };

  const res = sample;

  return (
    <div>
      <SectionHead
        eyebrow="Computer vision · API"
        title={
          <>
            Disease <span className="text-[var(--marigold)]">Diagnostic</span>
          </>
        }
        sub="Samples from /api/ui/vision/samples; diagnose via /api/vision/diagnose."
        right={<StatusChip tone="marigold">{note || "backend"}</StatusChip>}
      />
      {err && <div className="mb-3 text-[12px] font-mono2 text-[var(--chili)]">{err}</div>}

      <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-4">
        <div className="space-y-4">
          <Card ticks className="p-4">
            <span className="tick-b" />
            <div className="eyebrow mb-3">Field samples · API catalog</div>
            <div className="grid grid-cols-2 gap-2.5">
              {samples.map((s) => (
                <button
                  key={s.id}
                  onClick={() => diagnoseSample(s)}
                  className={`group relative rounded-[10px] overflow-hidden border text-left ${
                    sample?.id === s.id && !uploadUrl ? "border-[var(--gold)]" : "border-[var(--line2)]"
                  }`}
                >
                  <img src={s.img} alt={s.name} className="w-full h-[86px] object-cover" />
                  <div className="absolute inset-0 bg-gradient-to-t from-[#08130cd9] to-transparent" />
                  <div className="absolute bottom-1.5 left-2 right-2">
                    <div className="text-[11.5px] font-semibold">{s.name}</div>
                    <div className="text-[10px] devnagari text-[var(--mut)]">{s.nameMr}</div>
                  </div>
                </button>
              ))}
            </div>
          </Card>
          <Card className="p-4">
            <div className="eyebrow mb-3">Upload leaf photo</div>
            <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(e) => onUpload(e.target.files?.[0])} />
            <button className="btn btn-ghost w-full" onClick={() => fileRef.current?.click()}>
              📸 Browse image → API diagnose
            </button>
          </Card>
        </div>

        <Card ticks className="p-5">
          <span className="tick-b" />
          {phase === "scanning" && <div className="text-[var(--gold)] font-mono2 mb-3">Scanning…</div>}
          {res && (
            <>
              <div className="flex gap-4 flex-wrap">
                {(uploadUrl || res.img) && (
                  <img src={uploadUrl || res.img} alt="" className="w-[160px] h-[160px] object-cover rounded-[12px] border border-[var(--line)]" />
                )}
                <div className="min-w-0 flex-1">
                  <div className="font-display font-bold text-[22px]">{res.disease}</div>
                  <div className="devnagari text-[var(--mut)]">{res.diseaseMr}</div>
                  <div className="mt-2 font-mono2 text-[var(--gold)]">conf {(Number(res.conf) * 100 || 0).toFixed(0)}%</div>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3 text-[13px]">
                <div>
                  <div className="eyebrow mb-1">Symptoms</div>
                  <p className="text-[var(--ink)]">{res.symptomsEn}</p>
                  <p className="devnagari text-[var(--mut)] mt-1">{res.symptomsMr}</p>
                </div>
                <div>
                  <div className="eyebrow mb-1">Treatment</div>
                  <p className="text-[var(--ink)]">{res.chemEn || res.organicEn}</p>
                  <p className="devnagari text-[var(--mut)] mt-1">{res.chemMr || res.organicMr}</p>
                </div>
              </div>
            </>
          )}
          {!res && <div className="text-[var(--dim)]">Loading samples from API…</div>}
        </Card>
      </div>
    </div>
  );
}
