const jsonify = async (res) => {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return res.json();
};

const post = (url, body) =>
  fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  }).then(jsonify);

export const api = {
  health: () => fetch("/api/health").then(jsonify),
  tasks: () => fetch("/api/tasks").then(jsonify),
  models: () => fetch("/api/models").then(jsonify),
  benchmarks: () => fetch("/api/benchmarks").then(jsonify),
  simulations: () => fetch("/api/simulations").then(jsonify),
  simulation: (id) => fetch(`/api/simulation/${id}`).then(jsonify),
  startSimulation: (config) => post("/api/simulation/start", config),
  inference: (task, features) => post("/api/inference", { task, features }),
};
