import { useStream } from "@langchain/langgraph-sdk/react";
import { Button } from "./components/ui/button";
import { useQueryState } from "nuqs";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { PlusIcon, MessageCircle, LoaderCircle, ThumbsUp } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./components/ui/dialog";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { ThinkingBlock } from "./components/thinking";

function App() {
  const [threadId, setThreadId] = useQueryState("threadId");
  const state = useStream<
    {
      topic: string;
      sections: Array<{
        name: string;
        description: string;
        research: boolean;
        content: string;
      }>;
      completed_sections: Array<string>;
      final_report: string;
    },
    { InterruptType: string }
  >({
    apiUrl: "http://localhost:2024",
    assistantId: "open_deep_research",

    threadId,
    onThreadId: setThreadId,
  });

  return (
    <main className="grid min-h-screen grid-rows-[auto,1fr,auto]">
      <div className="sticky top-0 grid min-h-[68px] grid-cols-[1fr,auto] items-center gap-4 bg-background p-4">
        {threadId ? (
          <>
            <div className="flex items-center gap-2">
              <MessageCircle className="size-5" />
              <span className="flex items-baseline gap-2">
                Thread <code className="text-muted-foreground">{threadId}</code>
              </span>
            </div>
            <Button size="sm" onClick={() => setThreadId(null)}>
              <PlusIcon className="size-5" />
              New Thread
            </Button>
          </>
        ) : (
          <span />
        )}
      </div>

      <div className="flex flex-col items-center justify-center p-4">
        <div className="flex flex-col gap-4">
          {state.values.topic ? (
            <div className="flex items-center gap-1.5">
              {state.isLoading ? (
                <>
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                  <span className="text-muted-foreground">
                    Researching topic:{" "}
                  </span>
                  {state.values.topic}
                </>
              ) : (
                <>
                  <span className="text-muted-foreground">
                    Research for topic:{" "}
                  </span>
                  {state.values.topic}
                </>
              )}
            </div>
          ) : (
            <div>
              <strong className="font-medium">Open Deep Research</strong>
              {": "}
              <span className="text-muted-foreground">
                Enter a topic to research
              </span>
            </div>
          )}

          {state.history.length > 0 && (
            <ThinkingBlock>
              {state.history.map((h) => (
                <div key={h.checkpoint.checkpoint_id} className="empty:hidden">
                  {h.tasks
                    .filter((t) => t.name !== "__start__")
                    .map((t) => (
                      <div key={t.id}>
                        <span className="inline-flex rounded-lg border px-2 py-1 text-sm text-muted-foreground">
                          {t.name}
                        </span>
                        <pre className="text-sm">
                          {JSON.stringify(t.result, null, 2)}
                        </pre>
                      </div>
                    ))}
                </div>
              ))}
            </ThinkingBlock>
          )}

          {state.values.final_report && (
            <div className="prose break-words">
              <Markdown remarkPlugins={[remarkGfm]}>
                {state.values.final_report}
              </Markdown>
            </div>
          )}

          {state.error ? (
            <div className="text-destructive">
              <p>{JSON.stringify(state.error)}</p>
            </div>
          ) : null}

          {state.interrupt ? (
            <div className="flex max-w-[65ch] flex-col gap-4 rounded-2xl border p-4">
              <p className="font-semibold uppercase text-muted-foreground">
                Interrupted
              </p>
              <div className="prose whitespace-pre-wrap">
                {state.interrupt.value}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <Dialog>
                  <DialogTrigger asChild>
                    <Button variant="outline">
                      <MessageCircle className="size-4" />
                      Give feedback
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-[425px]">
                    <DialogHeader>
                      <DialogTitle>Give feedback</DialogTitle>
                      <DialogDescription>
                        Provide a short feedback about the research plan.
                      </DialogDescription>
                    </DialogHeader>
                    <form
                      className="flex flex-col gap-4"
                      onSubmit={(e) => {
                        e.preventDefault();
                        const form = e.target as HTMLFormElement;
                        const formData = new FormData(form);
                        const feedback = formData.get("feedback") as string;
                        form.reset();

                        state.submit(undefined, {
                          command: { resume: feedback },
                        });
                      }}
                    >
                      <div className="grid gap-2">
                        <Label htmlFor="feedback">Feedback</Label>
                        <Input id="feedback" name="feedback" />
                      </div>
                      <DialogFooter>
                        <Button type="submit">Submit feedback</Button>
                      </DialogFooter>
                    </form>
                  </DialogContent>
                </Dialog>

                <Button
                  className="w-full"
                  onClick={() =>
                    state.submit(undefined, { command: { resume: true } })
                  }
                >
                  <ThumbsUp className="size-4" />
                  Approve
                </Button>
              </div>
            </div>
          ) : null}
        </div>
      </div>

      <div className="sticky bottom-0 p-4">
        <form
          className="mx-auto flex max-w-[calc(65ch+32px)] flex-col gap-3 rounded-xl border border-input bg-background focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 focus-within:ring-offset-background"
          onSubmit={(e) => {
            e.preventDefault();
            const form = e.target as HTMLFormElement;
            const formData = new FormData(form);
            const topic = formData.get("topic") as string;
            form.reset();

            state.submit({ topic }, { optimisticValues: { topic } });
          }}
        >
          <input
            placeholder="Specify a topic..."
            className="border-none bg-transparent p-4 outline-none"
            type="text"
            name="topic"
            disabled={state.isLoading}
          />
          <div className="flex items-center justify-end gap-2 p-3 pt-0">
            {state.isLoading ? (
              <Button key="stop" type="button" onClick={() => state.stop()}>
                Stop
              </Button>
            ) : (
              <Button type="submit">Research</Button>
            )}
          </div>
        </form>
      </div>
    </main>
  );
}

export default App;
