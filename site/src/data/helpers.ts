import database from "./pasta_database.json";

export function getShapes() {
  return Object.values(database.shapes)
    .filter((s: any) => {
      // Must have description or history
      if (!s.description && !s.history) return false;
      // Name sanity
      if (!s.name || s.name.length > 50 || s.name.length < 3) return false;
      return true;
    })
    .sort((a: any, b: any) => a.name.localeCompare(b.name));
}

export function getShape(slug: string) {
  const shapes = database.shapes as Record<string, any>;
  return Object.values(shapes).find((s: any) => s.slug === slug);
}

export function getDoughs() {
  return Object.values(database.doughs)
    .sort((a: any, b: any) => {
      // Sort by whether they have ingredients (recipes first)
      const aHas = (a as any).ingredients?.length || 0;
      const bHas = (b as any).ingredients?.length || 0;
      return bHas - aHas;
    });
}

export function getDough(slug: string) {
  const doughs = database.doughs as Record<string, any>;
  return doughs[slug] || Object.values(doughs).find((d: any) => d.slug === slug);
}

export function getShapesForDough(doughSlug: string) {
  return getShapes().filter((s: any) =>
    s.dough_recipes?.includes(doughSlug) || s.used_for?.includes(doughSlug)
  );
}

export { database };
