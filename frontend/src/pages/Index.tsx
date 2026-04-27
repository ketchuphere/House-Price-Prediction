import { useState } from "react";
import { Home, Sparkles, BarChart3, Zap, ShieldCheck } from "lucide-react";
import { PredictionForm } from "@/components/predictor/PredictionForm";
import { PredictionResult } from "@/components/predictor/PredictionResult";
import { predictPrice, type PredictInput, type PredictResult } from "@/lib/predict";
import { toast } from "@/hooks/use-toast";

const Index = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);

  const handleSubmit = async (data: PredictInput) => {
    setLoading(true);
    try {
      const r = await predictPrice(data);
      setResult(r);
      setTimeout(() => {
        document.getElementById("result")?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch {
      toast({
        title: "Prediction failed",
        description: "Please try again in a moment.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-soft">
      {/* Decorative background blobs */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden -z-0">
        <div className="absolute -top-40 -right-32 h-96 w-96 rounded-full bg-primary/10 blur-3xl animate-float" />
        <div className="absolute top-1/2 -left-32 h-96 w-96 rounded-full bg-primary-glow/10 blur-3xl animate-float" style={{ animationDelay: "2s" }} />
      </div>

      <div className="relative z-10">
        {/* Header */}
        <header className="container mx-auto px-4 sm:px-6 pt-8 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-hero shadow-soft">
              <Home className="h-5 w-5 text-primary-foreground" />
            </div>
            <div>
              <p className="text-base font-bold tracking-tight text-foreground">Estimate</p>
              <p className="text-xs text-muted-foreground -mt-0.5">AI house pricing</p>
            </div>
          </div>
        </header>

        {/* Hero */}
        <section className="container mx-auto px-4 sm:px-6 pt-12 pb-10 text-center max-w-3xl">
          <div className="animate-fade-in-up inline-flex items-center gap-2 rounded-full border border-border bg-background/60 backdrop-blur px-4 py-1.5 text-xs font-medium text-muted-foreground shadow-soft">
            <Sparkles className="h-3.5 w-3.5 text-primary" />
            Powered by machine learning
          </div>
          <h1 className="animate-fade-in-up mt-6 text-4xl sm:text-6xl font-bold tracking-tight text-foreground">
            Know your home's{" "}
            <span className="bg-gradient-hero bg-clip-text text-transparent">true value</span>
          </h1>
          <p className="animate-fade-in-up mt-5 text-base sm:text-lg text-muted-foreground max-w-xl mx-auto">
            Get an instant, data-driven price estimate in seconds. Just tell us a few details about
            the property.
          </p>
        </section>

        {/* Form + Result */}
        <section className="container mx-auto px-4 sm:px-6 pb-16 max-w-2xl">
          <div className="animate-scale-in rounded-2xl border border-border bg-gradient-card p-6 sm:p-8 shadow-elegant">
            <h2 className="text-xl font-semibold text-foreground mb-1">Property details</h2>
            <p className="text-sm text-muted-foreground mb-6">
              All fields are required for the most accurate estimate.
            </p>
            <PredictionForm loading={loading} onSubmit={handleSubmit} />
          </div>

          <div id="result" className="mt-8 scroll-mt-8">
            {result && <PredictionResult result={result} />}
          </div>
        </section>

        {/* Features */}
        <section className="container mx-auto px-4 sm:px-6 pb-20 max-w-4xl">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {[
              { icon: Zap, title: "Instant results", desc: "Get a full estimate in under a second." },
              { icon: BarChart3, title: "Price ranges", desc: "See realistic low and high bounds." },
              { icon: ShieldCheck, title: "Confidence scoring", desc: "Know how reliable each estimate is." },
            ].map(({ icon: Icon, title, desc }) => (
              <div
                key={title}
                className="rounded-2xl border border-border bg-background/60 backdrop-blur p-5 shadow-soft transition-smooth hover:shadow-elegant hover:-translate-y-0.5"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent text-accent-foreground mb-3">
                  <Icon className="h-5 w-5" />
                </div>
                <p className="font-semibold text-foreground">{title}</p>
                <p className="mt-1 text-sm text-muted-foreground">{desc}</p>
              </div>
            ))}
          </div>
        </section>

        <footer className="container mx-auto px-4 sm:px-6 pb-8 text-center text-xs text-muted-foreground">
          © {new Date().getFullYear()} Estimate · Pricing estimates are informational only.
        </footer>
      </div>
    </div>
  );
};

export default Index;
