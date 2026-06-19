# DamageSim

To be updated

<!-- **DamageSim** is defined in **`damagesim/core`**: a portable, physics-informed **health** signal built from generic simulator measurements (forces, motion, temperature-related signals, contact summaries, and similar). Shared abstractions let you plug in different simulators by subclassing the same environment and mixin hooks; this page documents only that **core contract**, not any one engine.

!!! info "Relationship to OopsieBench"

    [OopsieBench](oopsiebench.md) wires these types into concrete task environments and scripts. For **which** tasks and **how** to run them, use that page.

---

## Mental model

1. Entities in the scene participate as **`DamageableMixin`** instances (**which** parts track health and **which** evaluators run are configuration concerns).
2. After each simulation step, **damage evaluators** read state attached to those entities and emit **per-part nonnegative damage**.
3. Damage is subtracted from **part health**; the optional **`obs["health"]`** vector and **`info["damage_info"]`** tree are **contracts** implemented on top of **`DamageableEnvironment`** and **`DamageableMixin`** in core.

**`DamageableEnvironment`** is mixed with a **simulator-native base environment** (multiple inheritance). The base class in core does not import a specific simulator; subclasses implement **`_get_all_objects()`**, **`step`/`reset` orchestration**, and **`_process_obs()`** as needed.

---

## Where to look (core)

| Module | Role |
|--------|------|
| [`damagesim/core/damageable_env.py`](https://github.com/UT-Austin-RobIn/oopsieverse/blob/main/damagesim/core/damageable_env.py) | **`DamageableEnvironment`**: tracking policy, **`reset`** / **`step`** helpers (**`_reset_damage_tracking`**, **`_update_all_health`**), **`health_list_link_names`**, **`_append_health_to_obs`**. |
| [`damagesim/core/damageable_mixin.py`](https://github.com/UT-Austin-RobIn/oopsieverse/blob/main/damagesim/core/damageable_mixin.py) | **`DamageableMixin`**: **`link_healths`**, evaluator wiring, **`update_health`**, **`damage_info`**. |
| [`damagesim/core/evaluators/base.py`](https://github.com/UT-Austin-RobIn/oopsieverse/blob/main/damagesim/core/evaluators/base.py) | **`DamageEvaluator`** ABC. |
| [`damagesim/core/evaluators/mechanical.py`](https://github.com/UT-Austin-RobIn/oopsieverse/blob/main/damagesim/core/evaluators/mechanical.py) | Mechanical damage math (impact + quasistatic contact); abstract hooks for simulator data. |
| [`damagesim/core/evaluators/thermal.py`](https://github.com/UT-Austin-RobIn/oopsieverse/blob/main/damagesim/core/evaluators/thermal.py) | Thermal evaluator base. |
| [`damagesim/core/evaluators/electrical.py`](https://github.com/UT-Austin-RobIn/oopsieverse/blob/main/damagesim/core/evaluators/electrical.py) | Electrical evaluator base. |

---

## 1. **`DamageableEnvironment`**

[**`DamageableEnvironment`**](https://github.com/UT-Austin-RobIn/oopsieverse/blob/main/damagesim/core/damageable_env.py) is the simulator-agnostic environment side of the stack. A concrete simulator integration typically declares something of the shape:

```text
class MyDamageableEnv(DamageableEnvironment, SimulatorBaseEnvironment):
    ...
```

The **simulator-specific** parent must still provide the real **`reset`**, **`step`**, and physics; **`DamageableEnvironment`** supplies reusable damage bookkeeping and observation helpers Subclasses extend.

### **`_get_all_objects()`** (abstract in spirit)

Core logic iterates **`self._get_all_objects()`**, which subclasses **must implement** to return scene entities that may carry **`DamageableMixin`** (objects, robots, fixtures, etc.). The core module never enumerates simulator types itself.

### Responsibilities owned by core

| Concern | Behavior in core |
|---------|-------------------|
| **Flags** | **`damage_config`** (e.g. **`enable_damage`**, **`track_robot_damage`**) and **`lock_health`** to freeze health updates without removing wrappers. |
| **Tracking YAML / dict** | **`damage_trackable_objects_config`** is a dict (often loaded by the subclass constructor). **`_load_damage_trackable_objects_config()`** defaults to **`{}`**; subclasses may override where that file lives. |
| **`initialize_damageable_objects()`** | For each **`DamageableMixin`** from **`_get_all_objects()`**, sets **`track_damage`**, invokes **`set_damageable_links_and_params()`**, **`initialize_health()`**, and evaluator setup according to allowlist/denylist rules below. |
| **`_reset_damage_tracking()`** | On reset: evaluator **`reset_tracking`**, re-initialize health, rebuild **`health_list_link_names`** (**`_build_health_list`**). |
| **`_initialize_all_evaluators()`** | Lazily constructs evaluators on first step if needed. |
| **`_update_all_health()`** | Runs **`update_health()`** on every tracked mixin; returns **`{ object_name: object.damage_info }`** (used as **`info["damage_info"]`** in typical subclasses). |
| **`_append_health_to_obs(obs)`** | Concatenates per-link health values into **`obs["health"]`**. |

**Discovery / class replacement** (swapping primitive objects for damageable subclasses, attaching default **`damage_params`**, etc.) is **not** in core—it lives in simulator packages that subclass **`DamageableEnvironment`**.

### Who gets tracked?

**`initialize_damageable_objects`** interprets **`damage_trackable_objects_config`** using **`task_name`** (from **`self.task_name`** or **`self._task_name`**):

1. **Task-specific key (allowlist)** — For any key other than **`"default"`** that matches **`task_name`**, an object is tracked if its **`category`** or **`name`** appears in configured sets (with substring matching between names and categories). The sentinel category **`"agent"`** can mark the controlling robot (**`_is_robot`** heuristic in core).
2. **`"default"` (denylist)** — Under the **`default`** entry, every **`DamageableMixin`** is tracked **except** categories listed in **`skip_categories`**.

If the config dict is empty or missing keys, nothing in core invents new rules: the subclass decides what **`task_name`** and merged config are before **`initialize_damageable_objects()`** runs.

---

## 2. **`DamageableMixin`**

[**`DamageableMixin`**](https://github.com/UT-Austin-RobIn/oopsieverse/blob/main/damagesim/core/damageable_mixin.py) adds **per-entity** health and evaluator state. It is meant to be composed with a simulator’s native object/robot class via multiple inheritance; the mixin should appear **before** that base in the MRO so cooperative **`__init__`** chains stay valid (see the docstring on the class).

### Terminology

- **Link** — Core’s name for the **atomic part** (one string key) whose health is tracked. A backend maps this to whatever the simulator calls a link, body, or geom group.
- **`damageable_links`** — Ordered list of part names with health.
- **`link_healths`** — **`part_name →`** scalar health (full = **100**).

**`health`** (property) is the **minimum** over **`link_healths`**, a single scalar summary.

### Configuration

**`damage_params`** is a plain dict (populated by the subclass or factory). Important keys:

| Key | Meaning |
|-----|---------|
| **`damage_evaluators`** | Names (strings) passed to **`_get_evaluator_registry()`** to instantiate evaluators. |
| Nested dicts per evaluator name | Keyword arguments for that evaluator class. |

**`set_damageable_links_and_params()`** is implemented above core (or in a thin backend mixin) to fill **`damageable_links`** and **`damage_params`** from categories, YAML, or overrides.

### Evaluator lifecycle

| Method | Role |
|--------|------|
| **`_get_evaluator_registry()`** | Must be implemented on the mixin subclass: returns **`name → EvaluatorClass`**. |
| **`_initialize_damage_evaluators()`** | Clears and rebuilds **`self.damage_evaluators`** from **`damage_params["damage_evaluators"]`**. |
| **`reset_damage_evaluators()`** | **`reset_tracking()`** on each evaluator (e.g. on env reset). |
| **`update_health()`** | Calls **`generate_damage()`** on each evaluator, subtracts damage from **`link_healths`**, fills **`damage_info`**. |

---

## 3. **`DamageEvaluator`**

[**`DamageEvaluator`**](https://github.com/UT-Austin-RobIn/oopsieverse/blob/main/damagesim/core/evaluators/base.py) binds to **`entity`** (the **`DamageableMixin`** instance). Subclasses implement:

```python
def generate_damage(self) -> Dict[str, float]:
    ...
```

returning nonnegative **damage per part name** for the current step. The base constructor exposes **`damage_threshold`** and **`scale`** (concrete evaluators may add aliases such as mechanical **`damage_scale`**).

### Families in core

| Module | Purpose |
|--------|---------|
| **`mechanical.py`** | Shared impact + quasistatic contact model; abstract **`_get_*`** hooks supply velocities, contacts, masses, and dt from the simulator. |
| **`thermal.py`** | Temperature-based damage when thresholds are exceeded. |
| **`electrical.py`** | Contact / particle-style exposure for electrical damage. |

Core evaluators **do not** import a physics engine. Subclasses in **`damagesim/<backend>/evaluators/`** implement the hooks and register concrete classes in the registry returned by **`DamageableMixin._get_evaluator_registry()`**.

---

## 4. Augmented **`step`**, **`health`**, **`health_list_link_names`**, **`damage_info`**

### Intended order of operations

The core type does not override **`step`** itself; the pattern it enables is:

1. Run the **simulator** **`step`** (physics, rewards, termination flags—defined by the base env).
2. Optionally, a subclass may **skip** **`_update_all_health()`** for a bounded warm-up (e.g. loading scripted states) so the first frames do not apply damage.
3. If **`lock_health`** is false, call **`_update_all_health()`** so every tracked **`DamageableMixin`** runs **`update_health()`**.
4. Put the returned mapping into **`info["damage_info"]`** (convention used by shipped integrations).
5. Call **`_process_obs(obs)`**; at minimum, core’s **`_append_health_to_obs`** adds **`obs["health"]`**. Further keys (proprio, cameras, etc.) are entirely subclass-defined.

**Gym-style return tuples** (e.g. terminated vs truncated vs done) depend on the **simulator base**, not on **`damagesim/core`**.

### **`obs["health"]`**

**`_append_health_to_obs`** walks **`_get_all_objects()`** in order; for each entity with **`track_damage`**, it appends **`link_healths[link]`** for each **`link`** in **`damageable_links`**, producing a **`numpy.float32`** 1‑D array. A subclass may cast or copy that array for framework consistency.

**`health[i]`** only lines up with a part if you also read **`health_list_link_names[i]`**.

### **`health_list_link_names`**

Rebuilt in **`_reset_damage_tracking`** via **`_build_health_list`**: strings **`"{object_name}@{link_name}"`**, in the **same** order as **`obs["health"]`** (same object iteration order, same **`damageable_links`** order per object). The attribute **`DamageableEnvironment.health_list_link_names`** holds the list for the current episode after reset.

### **`damage_info`**

Per object, **`DamageableMixin.update_health`** structures diagnostics as:

```text
damage_info[part_name][evaluator_name] = { ... }
```

Each leaf dict includes at least **`damage`** (amount applied this step). **`_record_evaluator_info`** in core adds evaluator-specific keys (e.g. mechanical forces and contacts, thermal temperature, electrical particle counts) so logging stays consistent across backends.

**`info["damage_info"]`** in **`step`/`reset`** is the dict **`{ object_name: that_object’s damage_info }`** assembled by the environment subclass—core’s **`_update_all_health`** returns exactly that mapping for convenience.

Use **`damage_info`** when you need interpretability; use **`health`** + **`health_list_link_names`** for a fixed-size vector observation.

---

## Packages outside this page’s contract

The **`damagesim/core`** module is authoritative for the abstraction; **other top-level directories under [`damagesim/`](https://github.com/UT-Austin-RobIn/oopsieverse/tree/main/damagesim)** ship concrete integrations (environment subclasses, mixins, parameters, evaluator hooks tied to particular engines). They follow the contracts on this page and add replacement factories, default configs, observation shaping, I/O metadata, and API details specific to each simulator.

When implementing a **new** backend, mirror **`DamageableEnvironment`** and **`DamageableMixin`** hook patterns from core rather than duplicating health math—then place the glue code in its own subdirectory next to **`core`**. -->
