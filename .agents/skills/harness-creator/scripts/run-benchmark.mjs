#!/usr/bin/env node
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  formatScoreReport,
  htmlReport,
  loadHarnessFiles,
  parseArgs,
  readJson,
  scoreHarness,
  writeText
} from './lib/harness-utils.mjs';

const args = parseArgs(process.argv.slice(2));
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const skillRoot = path.resolve(scriptDir, '..');

if (args.help) {
  console.log(`Usage: node scripts/run-benchmark.mjs [--target DIR] [--output FILE] [--html FILE]

Runs a lightweight harness benchmark:
  1. Scores the current target harness.
  2. Checks eval coverage in evals/evals.json.
  3. Produces a JSON report and optional HTML report.

This is a structural benchmark, not an LLM judge. Use it before/after real agent sessions.`);
  process.exit(0);
}

const target = path.resolve(args.target || args._[0] || process.cwd());
const output = path.resolve(args.output || path.join(target, 'harness-benchmark.json'));
const evalPath = path.resolve(args.evals || path.join(skillRoot, 'evals', 'evals.json'));

const harnessResult = scoreHarness(await loadHarnessFiles(target));
const evals = await readJson(evalPath);
const evalResult = scoreEvals(evals);
const report = {
  generatedAt: new Date().toISOString(),
  target,
  harness: harnessResult,
  evals: evalResult,
  recommendation: recommend(harnessResult, evalResult)
};

await writeText(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(`Benchmark report written to ${output}`);
console.log('');
console.log(formatScoreReport(harnessResult, target));
console.log(`Eval coverage: ${evalResult.score}/100 (${evalResult.passed}/${evalResult.total})`);
console.log(`Recommendation: ${report.recommendation}`);

if (args.html) {
  const htmlPath = path.resolve(args.html);
  await writeText(htmlPath, renderBenchmarkHtml(report));
  console.log(`HTML benchmark report written to ${htmlPath}`);
}

if (harnessResult.overall < Number(args.minScore || 70) || evalResult.score < Number(args.minEvalScore || 80)) {
  process.exitCode = 1;
}

function scoreEvals(evalsJson) {
  const cases = Array.isArray(evalsJson.evals) ? evalsJson.evals : [];
  const checks = [];
  checks.push({ pass: cases.length >= 10, message: 'At least 10 eval cases' });
  checks.push({ pass: cases.some((item) => /minimal|creation/i.test(item.name)), message: 'Covers minimal harness creation' });
  checks.push({ pass: cases.some((item) => /session|continuity/i.test(item.name)), message: 'Covers session continuity' });
  checks.push({ pass: cases.some((item) => /assessment|score/i.test(item.name)), message: 'Covers harness assessment' });
  checks.push({ pass: cases.some((item) => /verification/i.test(item.name)), message: 'Covers verification workflow' });
  checks.push({ pass: cases.some((item) => /memory/i.test(item.name)), message: 'Covers memory taxonomy' });
  checks.push({ pass: cases.some((item) => /tool|permission|safety/i.test(item.name)), message: 'Covers tool safety' });
  checks.push({ pass: cases.some((item) => /multi-agent|delegation|coordination/i.test(item.name)), message: 'Covers multi-agent coordination' });
  checks.push({ pass: cases.every((item) => item.prompt && item.expected_output && Array.isArray(item.expectations)), message: 'Each eval has prompt, expected output, expectations' });
  checks.push({ pass: cases.every((item) => item.expectations?.length >= 3), message: 'Each eval has at least three expectation checks' });

  const passed = checks.filter((check) => check.pass).length;
  return {
    score: Math.round((passed / checks.length) * 100),
    passed,
    total: checks.length,
    cases: cases.length,
    checks
  };
}

function recommend(harnessResult, evalResult) {
  if (harnessResult.overall >= 85 && evalResult.score >= 90) {
    return 'Ready for realistic before/after agent-session benchmarking.';
  }
  if (harnessResult.overall < 70) {
    return `Improve the ${harnessResult.bottleneck} subsystem before benchmarking agent behavior.`;
  }
  if (evalResult.score < 80) {
    return 'Expand eval coverage before treating benchmark results as representative.';
  }
  return 'Usable, with some gaps worth tightening after first real sessions.';
}

function renderBenchmarkHtml(report) {
  const evalHtml = htmlReport(report.harness, `Harness Benchmark: ${path.basename(report.target)}`)
    .replace('</main>', `<section>
      <h2>Eval Coverage <span>${report.evals.score}/100</span></h2>
      <p>${report.evals.passed}/${report.evals.total} benchmark checks passed across ${report.evals.cases} eval cases.</p>
      <ul>${report.evals.checks.map((check) => `<li class="${check.pass ? 'pass' : 'fail'}">${check.pass ? 'PASS' : 'FAIL'} ${escapeHtml(check.message)}</li>`).join('')}</ul>
    </section>
    <section>
      <h2>Recommendation</h2>
      <p>${escapeHtml(report.recommendation)}</p>
    </section>
  </main>`);
  return evalHtml;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}
