import { useState } from "react";
import { Bath, BedDouble, Calendar, Loader2, MapPin, Ruler, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { predictSchema, type PredictInput } from "@/lib/predict";
import { cn } from "@/lib/utils";

type Props = {
  loading: boolean;
  onSubmit: (data: PredictInput) => void;
};

const initial = {
  location: "",
  squareFeet: "",
  bedrooms: "",
  bathrooms: "",
  yearBuilt: "",
};

type Errors = Partial<Record<keyof typeof initial, string>>;

export function PredictionForm({ loading, onSubmit }: Props) {
  const [values, setValues] = useState(initial);
  const [errors, setErrors] = useState<Errors>({});

  const update = (key: keyof typeof initial, v: string) => {
    setValues((s) => ({ ...s, [key]: v }));
    if (errors[key]) setErrors((e) => ({ ...e, [key]: undefined }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const parsed = predictSchema.safeParse({
      location: values.location,
      squareFeet: values.squareFeet === "" ? NaN : Number(values.squareFeet),
      bedrooms: values.bedrooms === "" ? NaN : Number(values.bedrooms),
      bathrooms: values.bathrooms === "" ? NaN : Number(values.bathrooms),
      yearBuilt: values.yearBuilt === "" ? NaN : Number(values.yearBuilt),
    });
    if (!parsed.success) {
      const next: Errors = {};
      for (const issue of parsed.error.issues) {
        const k = issue.path[0] as keyof typeof initial;
        if (!next[k]) next[k] = issue.message;
      }
      setErrors(next);
      return;
    }
    onSubmit(parsed.data);
  };

  const fields = [
    {
      key: "location" as const,
      label: "Location",
      icon: MapPin,
      placeholder: "e.g. Austin, TX",
      type: "text",
      span: "sm:col-span-2",
    },
    {
      key: "squareFeet" as const,
      label: "Square Feet",
      icon: Ruler,
      placeholder: "1,800",
      type: "number",
    },
    {
      key: "yearBuilt" as const,
      label: "Year Built",
      icon: Calendar,
      placeholder: "2005",
      type: "number",
    },
    {
      key: "bedrooms" as const,
      label: "Bedrooms",
      icon: BedDouble,
      placeholder: "3",
      type: "number",
    },
    {
      key: "bathrooms" as const,
      label: "Bathrooms",
      icon: Bath,
      placeholder: "2",
      type: "number",
    },
  ];

  return (
    <form onSubmit={handleSubmit} className="grid grid-cols-1 sm:grid-cols-2 gap-5">
      {fields.map(({ key, label, icon: Icon, placeholder, type, span }) => (
        <div key={key} className={cn("space-y-2", span)}>
          <Label htmlFor={key} className="text-sm font-medium text-foreground/80">
            {label}
          </Label>
          <div className="relative">
            <Icon className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
            <Input
              id={key}
              type={type}
              inputMode={type === "number" ? "decimal" : "text"}
              placeholder={placeholder}
              value={values[key]}
              onChange={(e) => update(key, e.target.value)}
              maxLength={type === "text" ? 100 : undefined}
              className={cn(
                "pl-10 h-12 rounded-xl border-border bg-background/60 backdrop-blur transition-smooth",
                "focus-visible:ring-2 focus-visible:ring-ring focus-visible:border-ring",
                errors[key] && "border-destructive focus-visible:ring-destructive",
              )}
              aria-invalid={Boolean(errors[key])}
              aria-describedby={errors[key] ? `${key}-error` : undefined}
            />
          </div>
          {errors[key] && (
            <p id={`${key}-error`} className="text-xs text-destructive">
              {errors[key]}
            </p>
          )}
        </div>
      ))}

      <div className="sm:col-span-2 pt-2">
        <Button
          type="submit"
          disabled={loading}
          className={cn(
            "w-full h-12 rounded-xl text-base font-semibold",
            "bg-gradient-hero text-primary-foreground shadow-elegant hover:shadow-glow",
            "transition-smooth hover:scale-[1.01] active:scale-[0.99]",
          )}
        >
          {loading ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Predicting price...
            </>
          ) : (
            <>
              <Sparkles className="mr-2 h-5 w-5" />
              Predict Price
            </>
          )}
        </Button>
      </div>
    </form>
  );
}