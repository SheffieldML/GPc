/* CIP-0016 Phase A1: tests for the C++ interchange reader/writer.
 *
 * Usage: testInterchange [gp-scuderia/examples directory]
 * The default directory is ../gp-scuderia/examples. The goldens carry an
 * x-reference block produced by GPy (the numeric oracle); a missing directory is
 * a failure, not a skip. The analytic and negative cases need no files. */
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "CInterchange.h"
#include "CJson.h"
#include "ndlexceptions.h"

using namespace std;

namespace {

int failures = 0;

void expect(bool cond, const string& label)
{
  if(cond)
    cout << "PASS: " << label << endl;
  else
  {
    cout << "FAIL: " << label << endl;
    failures++;
  }
}

void expectNear(double got, double want, double tol, const string& label)
{
  if(std::fabs(got - want) <= tol)
    cout << "PASS: " << label << endl;
  else
  {
    cout << "FAIL: " << label << " (got " << got << ", want " << want << ")" << endl;
    failures++;
  }
}

// Returns true when f throws an ndlexceptions::Error whose message contains frag.
template <class F>
bool throwsWith(F f, const string& frag)
{
  try
  {
    f();
  }
  catch(ndlexceptions::Error& e)
  {
    return e.getMessage().find(frag) != string::npos;
  }
  return false;
}

const char* kAnalytic = R"({
  "schema_version": "0.1",
  "type": "models.gp_regression",
  "input_dim": 1,
  "output_dim": 1,
  "kernel": {"type": "kern.rbf", "variance": 1.0, "lengthscale": 1.0},
  "likelihood": {"type": "likelihoods.gaussian", "variance": 0.5},
  "data": {"X": [[0.0]], "Y": [[1.0]]},
  "x-note": "extension fields must survive a round trip"
})";

void testJson()
{
  CJson j = CJson::parse(R"({"a":[1,2.5,-3e2],"s":"x\n\u00e9","t":true,"n":null,"o":{}})");
  expect(j.at("a").size() == 3 && j.at("a").at(2).asNumber() == -300.0, "json: numbers and arrays");
  expect(j.at("s").asString() == "x\n\xC3\xA9", "json: escapes and utf-8");
  expect(j.at("t").asBool() && j.at("n").isNull() && j.at("o").size() == 0, "json: literals");
  CJson k = CJson::parse(j.dump());
  expect(k.dump(-1) == j.dump(-1), "json: dump/parse round trip");
  const double third = 1.0 / 3.0;
  CJson n = CJson::makeNumber(third);
  expect(CJson::parse(n.dump(-1)).asNumber() == third, "json: double round-trips exactly");
  expect(throwsWith([] { CJson::parse("{\"a\":1,}"); }, "JSON parse error"), "json: trailing comma rejected");
  expect(throwsWith([] { CJson::parse("[1,2"); }, "unterminated"), "json: truncated array rejected");
  expect(throwsWith([] { CJson::parse("{\"a\":1,\"a\":2}"); }, "duplicate key"), "json: duplicate key rejected");
  expect(throwsWith([] { CJson::parse("1 2"); }, "trailing"), "json: trailing garbage rejected");
}

void testAnalytic()
{
  auto m = CInterchange::fromJson(CJson::parse(kAnalytic));
  const double want = -0.5 * (std::log(2.0 * M_PI * 1.5) + 1.0 / 1.5);
  expectNear(m->logLikelihood(), want, 1e-12, "analytic: one-point log likelihood");
  CMatrix Xnew(1, 1);
  Xnew.setVal(0.0, 0, 0);
  vector<double> mean, var;
  m->predict(Xnew, mean, var);
  expectNear(mean[0], 1.0 / 1.5, 1e-12, "analytic: predictive mean");
  expectNear(var[0], 1.0 - 1.0 / 1.5 + 0.5, 1e-12, "analytic: predictive variance includes noise");

  CJson out = m->toJson();
  expect(out.has("x-note"), "round trip: extension field preserved");
  auto m2 = CInterchange::fromJson(out);
  expectNear(m2->logLikelihood(), want, 1e-12, "round trip: same log likelihood");

  // lengthscale <-> inverseWidth must invert exactly.
  CJson d = CJson::parse(kAnalytic);
  d.at("kernel").set("lengthscale", CJson::makeNumber(0.37));
  expectNear(CInterchange::fromJson(d)->toJson().at("kernel").at("lengthscale").asNumber(), 0.37,
             1e-13, "round trip: lengthscale");
}

void testWhiteIsAdditive()
{
  // kern.white 0.2 plus likelihood 0.3 must act as noise 0.5 on the diagonal.
  CJson d = CJson::parse(kAnalytic);
  CJson parts = CJson::makeArray();
  parts.push(d.at("kernel"));
  CJson w = CJson::makeObject();
  w.set("type", CJson::makeString("kern.white"));
  w.set("variance", CJson::makeNumber(0.2));
  parts.push(w);
  CJson sum = CJson::makeObject();
  sum.set("type", CJson::makeString("kern.add"));
  sum.set("parts", parts);
  d.set("kernel", sum);
  d.at("likelihood").set("variance", CJson::makeNumber(0.3));
  auto m = CInterchange::fromJson(d);
  const double want = -0.5 * (std::log(2.0 * M_PI * 1.5) + 1.0 / 1.5);
  expectNear(m->logLikelihood(), want, 1e-12, "white + likelihood both add to the Gram diagonal");
}

void testNegative()
{
  auto mutate = [](const string& key, const CJson& v) {
    CJson d = CJson::parse(kAnalytic);
    d.set(key, v);
    return d;
  };
  expect(throwsWith([&] { CInterchange::fromJson(mutate("schema_version", CJson::makeString("9.9"))); },
                    "schema_version"), "negative: wrong schema version");
  expect(throwsWith([&] { CInterchange::fromJson(mutate("type", CJson::makeString("models.nope"))); },
                    "unknown model type"), "negative: unknown model type");
  expect(throwsWith([&] { CInterchange::fromJson(mutate("input_dim", CJson::makeNumber(2))); },
                    "input_dim"), "negative: input_dim disagrees with data");
  {
    CJson d = CJson::parse(kAnalytic);
    d.erase("data");
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "data"), "negative: missing data");
  }
  {
    CJson d = CJson::parse(kAnalytic);
    d.at("kernel").set("type", CJson::makeString("kern.mystery"));
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "unknown kernel type"),
           "negative: unknown kernel type");
  }
  {
    CJson d = CJson::parse(kAnalytic);
    d.set("mean", CJson::parse("{\"type\": \"mean.constant\", \"value\": 1.0}"));
    d.set("normalizer", CJson::parse("{\"type\": \"normalizer.center\", \"mean\": 1.0}"));
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "cannot both be present"),
           "negative: mean together with normalizer");
  }
  {
    CJson d = CJson::parse(kAnalytic);
    d.set("normalizer", CJson::parse("{\"type\": \"normalizer.standardize\", \"mean\": 0.0, \"std\": 0.0}"));
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "must be positive"),
           "negative: non-positive normalizer std");
  }
  {
    CJson d = CJson::parse(kAnalytic);
    d.set("mean", CJson::parse("{\"type\": \"mean.linear\"}"));
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "only mean.constant"),
           "negative: unsupported mean type");
  }
  {
    CJson d = CJson::parse(kAnalytic);
    d.at("kernel").erase("lengthscale");
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "lengthscale"),
           "negative: missing lengthscale (truncated parameters)");
  }
  {
    CJson d = CJson::parse(kAnalytic);
    CJson ls = CJson::makeArray();
    ls.push(CJson::makeNumber(1.0));
    ls.push(CJson::makeNumber(2.0));
    d.at("kernel").set("lengthscale", ls);
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "input_dim"),
           "negative: ARD length disagrees with input_dim");
  }
  {
    CJson d = CJson::parse(kAnalytic);
    d.at("kernel").set("type", CJson::makeString("kern.prod"));
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "parts"), "negative: kern.prod without parts");
  }
  {
    CJson d = CJson::parse(kAnalytic);
    CJson prod = CJson::makeObject();
    prod.set("type", CJson::makeString("kern.prod"));
    CJson parts = CJson::makeArray();
    parts.push(d.at("kernel"));
    CJson w = CJson::makeObject();
    w.set("type", CJson::makeString("kern.white"));
    w.set("variance", CJson::makeNumber(0.1));
    parts.push(w);
    prod.set("parts", parts);
    d.set("kernel", prod);
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "kern.white"), "negative: kern.white inside kern.prod");
  }
  {
    CJson d = CJson::parse(kAnalytic);
    d.at("kernel").set("type", CJson::makeString("kern.matern52"));
    d.at("kernel").set("lengthscale", CJson::parse("[1.0]"));
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "ARD"), "negative: ARD matern52 refused");
  }
  {
    CJson d = CJson::parse(kAnalytic);
    d.at("likelihood").set("variance", CJson::makeNumber(-1.0));
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "positive"), "negative: non-positive variance");
  }
}

void testModelStreamRoundTrip()
{
  // A bias variance of exactly 0.5 is written as the hex float 0x1p-1, which the
  // matrix reader once turned into 0 (CIP-0016 A2 finding).
  CJson d = CJson::parse(kAnalytic);
  CJson parts = CJson::makeArray();
  parts.push(d.at("kernel"));
  CJson b = CJson::makeObject();
  b.set("type", CJson::makeString("kern.bias"));
  b.set("variance", CJson::makeNumber(0.5));
  parts.push(b);
  CJson sum = CJson::makeObject();
  sum.set("type", CJson::makeString("kern.add"));
  sum.set("parts", parts);
  d.set("kernel", sum);
  auto m = CInterchange::fromJson(d);
  const string path = "testInterchange_tmp.model";
  m->writeGpModel(path);
  CGp* g = readGpFromFile(path, 0);
  CMatrix Xn = m->inputs(), Yn = m->targets();
  g->setTargetVals(Yn);
  g->updateM();
  g->pX = &Xn;
  expectNear(-g->computeObjectiveVal(), m->logLikelihood(), 1e-12, ".model round trip keeps bias variance 0.5");
  remove(path.c_str());
  delete g;
}

void testProdMatern52ModelStream(const string& golden)
{
  // A product kernel and Matern 5/2 survive a .model write, read and export.
  auto m = CInterchange::fromFile(golden);
  const string path = "testInterchange_prod.model";
  m->writeGpModel(path);
  CGp* g = readGpFromFile(path, 0);
  CMatrix Xn = m->inputs(), Yn = m->targets();
  g->setTargetVals(Yn);
  g->updateM();
  g->pX = &Xn;
  CJson out = CInterchange::exportModel(*g);
  expect(out.at("kernel").dump().find("kern.prod") != string::npos
         && out.at("kernel").dump().find("kern.matern52") != string::npos,
         "prod and matern52 exported from a .model");
  remove(path.c_str());
  delete g;
}

void testCoregionalisation(const string& icm, const string& lcm)
{
  // ICM/LCM with per-output noise: .model round trip keeps active_dims, and bad
  // documents are refused with the offending field named.
  auto m = CInterchange::fromFile(lcm);
  const string path = "testInterchange_lcm.model";
  m->writeGpModel(path);
  CGp* g = readGpFromFile(path, 0);
  CMatrix Xn = m->inputs(), Yn = m->targets();
  g->setTargetVals(Yn);
  g->updateM();
  g->pX = &Xn;
  CJson out = CInterchange::exportModel(*g);  // verifies the log likelihood internally
  const string k = out.at("kernel").dump();
  expect(k.find("kern.coregionalize") != string::npos && k.find("active_dims") != string::npos,
         "coregionalize and active_dims exported from a .model");
  expect(out.at("likelihood").at("type").asString() == "likelihoods.mixed_gaussian",
         "index-white noise exported as likelihoods.mixed_gaussian");
  delete g;
  remove(path.c_str());

  const CJson base = CJson::parseFile(icm);
  {
    CJson d = base;
    d.at("kernel").at("parts").at(1).erase("active_dims");
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "active_dims"),
           "negative: coregionalize without active_dims");
  }
  {
    CJson d = base;
    d.at("kernel").at("parts").at(0).set("active_dims", CJson::parse("[7]"));
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "outside"), "negative: active_dims column out of range");
  }
  {
    CJson d = base;
    d.at("likelihood").erase("output_index_column");
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "output_index_column"),
           "negative: mixed_gaussian without output_index_column");
  }
  {
    CJson d = base;
    d.at("data").set("X", CJson::parse("[[0,5],[1,0],[2,0],[0.5,1],[1.5,1],[2.5,1]]"));
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "output index"),
           "negative: data output index outside the noise vector");
  }
  {
    CJson d = base;
    d.at("likelihood").set("variance", CJson::parse("[0.1]"));
    d.at("data").set("X", CJson::parse("[[0,0],[1,0],[2,0],[0.5,0],[1.5,0],[2.5,0]]"));
    d.at("kernel").at("parts").at(1).set("output_dim", CJson::makeNumber(1));
    expect(throwsWith([&] { CInterchange::fromJson(d); }, "rows"),
           "negative: W rows disagree with output_dim");
  }
}

void testWhiteModelStream(const string& golden, const string& noWhiteGolden)
{
  // CIP-0027 allow-white flag survives a .model write/read, so an exact GP with
  // kern.white round-trips; streams without the key still read as flag = false.
  auto m = CInterchange::fromFile(golden);
  const string path = "testInterchange_white.model";
  m->writeGpModel(path);
  CGp* g = readGpFromFile(path, 0);
  expect(g->isAllowWhiteAndLikelihoodNoise(), "allow-white flag read back from the .model");
  CMatrix Xn = m->inputs(), Yn = m->targets();
  g->setTargetVals(Yn);
  g->updateM();
  g->pX = &Xn;
  CJson out = CInterchange::exportModel(*g);  // verifies the log likelihood internally
  expect(out.at("kernel").dump().find("kern.white") != string::npos, "kern.white exported from a .model");
  g->setAllowWhiteAndLikelihoodNoise(false);
  expect(throwsWith([&] { CInterchange::exportModel(*g); }, "allow-white"),
         "negative: White without the allow-white flag is not exported");
  delete g;
  remove(path.c_str());

  auto n = CInterchange::fromFile(noWhiteGolden);
  n->writeGpModel(path);
  CGp* h = readGpFromFile(path, 0);
  expect(!h->isAllowWhiteAndLikelihoodNoise(), "streams without the key read as flag = false");
  delete h;
  remove(path.c_str());
}

void testLaplaceModelStream(const string& golden)
{
  // A Laplace model survives .model write, read and export: the stream records
  // laplaceInference, and the export reproduces likelihood, inference and kernel.
  auto m = CInterchange::fromFile(golden);
  const string path = "testInterchange_laplace.model";
  m->writeGpModel(path);
  CGp* g = readGpFromFile(path, 0);
  expect(g->isLaplaceInference(), golden + ": laplaceInference read back from the .model");
  CMatrix Xn = m->inputs(), Yn = m->targets();
  g->setTargetVals(Yn);
  g->updateM();
  g->pX = &Xn;
  expectNear(-g->computeObjectiveVal(), m->logLikelihood(), 1e-9, golden + ": .model reloads to the same log likelihood");
  CJson out = CInterchange::exportModel(*g);  // verifies the log likelihood internally
  const CJson& src = CJson::parseFile(golden);
  expect(out.at("type").asString() == "models.gp" && out.at("inference").asString() == "laplace"
         && out.at("likelihood").dump() == src.at("likelihood").dump(),
         golden + ": exported as models.gp with the same likelihood and inference");
  // GPc keeps the top level as a compound kernel, so a single leaf exports as a one-part kern.add.
  const CJson &ko0 = out.at("kernel"), &ks = src.at("kernel");
  const CJson& ko = (ko0.at("type").asString() == "kern.add" && ko0.at("parts").size() == 1)
      ? ko0.at("parts").at(0) : ko0;
  expect(ko.at("type").asString() == ks.at("type").asString()
         && std::fabs(ko.at("variance").asNumber() - ks.at("variance").asNumber()) < 1e-12
         && std::fabs(ko.at("lengthscale").asNumber() - ks.at("lengthscale").asNumber()) < 1e-12,
         golden + ": exported kernel matches");
  // A Gaussian stream carries no laplaceInference key.
  remove(path.c_str());
  delete g;
}

void checkGolden(const string& path)
{
  auto m = CInterchange::fromFile(path);
  const CJson& doc = CJson::parseFile(path);
  const CJson& ref = doc.at("x-reference");
  const CJson& tol = ref.at("tolerance");
  expectNear(m->logLikelihood(), ref.at("log_likelihood").asNumber(),
             tol.at("log_likelihood").asNumber(), path + ": log likelihood vs oracle");
  const CJson& xn = ref.at("Xnew");
  CMatrix Xnew(static_cast<unsigned int>(xn.size()), static_cast<unsigned int>(xn.at(0).size()));
  for(size_t i = 0; i < xn.size(); ++i)
    for(size_t j = 0; j < xn.at(0).size(); ++j)
      Xnew.setVal(xn.at(i).at(j).asNumber(), static_cast<unsigned int>(i), static_cast<unsigned int>(j));
  vector<double> mean, var;
  m->predict(Xnew, mean, var);
  // Regression: FITC predict once overwrote scaledM and changed the objective.
  expectNear(m->logLikelihood(), ref.at("log_likelihood").asNumber(),
             tol.at("log_likelihood").asNumber(), path + ": log likelihood unchanged by predict");
  const CJson& rm = ref.at("mean");
  const CJson& rv = ref.at("var");
  bool okMean = mean.size() == rm.size(), okVar = var.size() == rv.size();
  for(size_t i = 0; okMean && okVar && i < mean.size(); ++i)
  {
    okMean = okMean && std::fabs(mean[i] - rm.at(i).asNumber()) <= tol.at("mean").asNumber();
    okVar = okVar && std::fabs(var[i] - rv.at(i).asNumber()) <= tol.at("var").asNumber();
  }
  if(ref.has("latent_mean"))
  {
    vector<double> mu, s2;
    m->predictLatent(Xnew, mu, s2);
    bool okMu = mu.size() == ref.at("latent_mean").size(), okS2 = okMu;
    for(size_t i = 0; okMu && i < mu.size(); ++i)
    {
      okMu = okMu && std::fabs(mu[i] - ref.at("latent_mean").at(i).asNumber()) <= tol.at("mean").asNumber();
      okS2 = okS2 && std::fabs(s2[i] - ref.at("latent_var").at(i).asNumber()) <= tol.at("var").asNumber();
    }
    expect(okMu, path + ": latent predictive mean vs oracle");
    expect(okS2, path + ": latent predictive variance vs oracle");
  }
  expect(okMean, path + ": predictive mean vs oracle");
  expect(okVar, path + ": predictive variance vs oracle");

  // Writing back and re-reading must preserve the numbers.
  auto m2 = CInterchange::fromJson(m->toJson());
  expectNear(m2->logLikelihood(), m->logLikelihood(), 1e-12, path + ": write/read round trip");
}

}  // namespace

void testYamlReading(const string& dir)
{
  const char* names[] = {"ftc_rbf_white", "dtcvar_ard_data", "ftc_prod_matern52", "fitc_matern52",
                         "ftc_icm", "ftc_lcm", "ftc_standardize", "laplace_bernoulli", "laplace_poisson"};
  for(const char* n : names)
  {
    const string y = dir + "/" + n + ".yaml", j = dir + "/" + n + ".json";
#ifdef GPC_YAML
    expect(CJson::parseFile(y).dump(-1) == CJson::parseFile(j).dump(-1), string(n) + ": YAML parses to the same document as its JSON twin");
    auto a = CInterchange::fromFile(y), b = CInterchange::fromFile(j);
    expectNear(a->logLikelihood(), b->logLikelihood(), 1e-12, string(n) + ": YAML and JSON models agree");
#else
    bool refused = false;
    try { CJson::parseFile(y); } catch(ndlexceptions::Error& e) {
      refused = string(e.getMessage()).find("YAML=1") != string::npos;
    }
    expect(refused, string(n) + ": YAML refused with a rebuild hint when built without GPC_YAML");
#endif
  }
}

int main(int argc, char** argv)
{
  const string dir = argc > 1 ? argv[1] : "../gp-scuderia/examples";
  try
  {
    testJson();
    testAnalytic();
    testWhiteIsAdditive();
    testNegative();
    testModelStreamRoundTrip();
    checkGolden(dir + "/ftc_rbf_white.json");
    checkGolden(dir + "/dtcvar_ard_data.json");
    checkGolden(dir + "/ftc_prod_matern52.json");
    checkGolden(dir + "/fitc_matern52.json");
    testProdMatern52ModelStream(dir + "/ftc_prod_matern52.json");
    testWhiteModelStream(dir + "/ftc_rbf_white.json", dir + "/ftc_standardize.json");
    checkGolden(dir + "/ftc_icm.json");
    checkGolden(dir + "/ftc_lcm.json");
    testCoregionalisation(dir + "/ftc_icm.json", dir + "/ftc_lcm.json");
    checkGolden(dir + "/laplace_bernoulli.json");
    checkGolden(dir + "/laplace_poisson.json");
    testLaplaceModelStream(dir + "/laplace_bernoulli.json");
    testLaplaceModelStream(dir + "/laplace_poisson.json");
    testYamlReading(dir);
  }
  catch(ndlexceptions::Error& e)
  {
    cout << "FAIL: unexpected exception: " << e.getMessage() << endl;
    failures++;
  }
  cout << (failures ? "FAILED" : "ALL PASSED") << " (" << failures << " failures)" << endl;
  return failures ? 1 : 0;
}
