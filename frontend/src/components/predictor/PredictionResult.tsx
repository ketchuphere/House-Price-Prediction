import { useEffect, useState } from "react";
import { TrendingUp, ShieldCheck, ArrowDown, ArrowUp } from "lucide-react";
import type { PredictResult } from "@/lib/predict";
import { cn } from "@/lib/utils";

const formatCurrency = (n: number) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(n);

function useCountUp(target: number, duration = 900) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const from = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(Math.round(from + (target - from) * eased));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return value;
}

export function PredictionResult({ result }: { result: PredictResult }) {
  const animated = useCountUp(result.price);
  const confidencePct = Math.round(result.confidence * 100);
  const range = result.high - result.low;
  const pointer = ((result.price - result.low) / Math.max(range, 1)) * 100;

  const confidenceLabel =
    confidencePct >= 85 ? "High confidence" : confidencePct >= 70 ? "Moderate confidence" : "Low confidence";
  const confidenceColor =
    confidencePct >= 85
      ? "text-success"
      : confidencePct >= 70
      ? "text-primary"
      : "text-destructive";

  return (
    <div className="animate-scale-in space-y-6 rounded-2xl border border-border bg-gradient-card p-6 sm:p-8 shadow-elegant">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full bg-accent px-3 py-1 text-xs font-medium text-accent-foreground">
            <TrendingUp className="h-3.5 w-3.5" />
            Estimated value
          </div>
          <p className="mt-3 text-4xl sm:text-5xl font-bold tracking-tight text-foreground">
            {formatCurrency(animated)}
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            Based on comparable properties and market trends
          </p>
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium text-foreground/80">Predicted price range</span>
          <span className="text-muted-foreground">±{formatCurrency(Math.round(range / 2))}</span>
        </div>

        <div className="relative h-3 rounded-full bg-secondary overflow-hidden">
          <div className="absolute inset-y-0 left-0 right-0 bg-gradient-hero opacity-90" />
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2"
            style={{ left: `${Math.max(2, Math.min(98, pointer))}%` }}
          >
            <div className="h-5 w-5 rounded-full bg-background border-2 border-primary shadow-elegant" />
          </div>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="inline-flex items-center gap-1 text-muted-foreground">
            <ArrowDown className="h-3.5 w-3.5" />
            {formatCurrency(result.low)}
          </span>
          <span className="inline-flex items-center gap-1 text-muted-foreground">
            {formatCurrency(result.high)}
            <ArrowUp className="h-3.5 w-3.5" />
          </span>
        </div>
      </div>

      <div className="rounded-xl bg-secondary/60 p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="inline-flex items-center gap-2 text-sm font-medium text-foreground/80">
            <ShieldCheck className={cn("h-4 w-4", confidenceColor)} />
            Confidence
          </div>
          <span className={cn("text-sm font-semibold", confidenceColor)}>
            {confidencePct}% · {confidenceLabel}
          </span>
        </div>
        <div className="h-2 rounded-full bg-background overflow-hidden">
          <div
            className="h-full bg-gradient-hero transition-all duration-700 ease-smooth"
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>
    </div>
  );
}