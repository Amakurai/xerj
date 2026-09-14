export default async function globalTeardown() {
  const node = globalThis.__xerjNode;
  if (node) node.stop();
}
