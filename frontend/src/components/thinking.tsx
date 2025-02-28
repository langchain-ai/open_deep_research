import { useState } from "react";
import { cn } from "@/lib/utils";
import { ChevronRight, BrainIcon } from "lucide-react";

export const ThinkingBlock = ({ children }: { children: React.ReactNode }) => {
  const [open, setOpen] = useState(false);

  return (
    <div
      onClick={!open ? () => setOpen(true) : undefined}
      className={cn(
        "flex max-w-[65ch] flex-col gap-4 overflow-auto rounded-xl border p-4",
        !open && "max-h-[30ch] overflow-hidden",
      )}
    >
      <button
        type="button"
        className="grid grid-cols-[auto,1fr,auto] items-center gap-3"
        onClick={() => setOpen(!open)}
      >
        <BrainIcon className="size-5" />
        <span className="text-left">Thinking</span>
        <ChevronRight
          className={cn("size-5 transition-all", open && "rotate-90")}
        />
      </button>
      {children}
    </div>
  );
};
