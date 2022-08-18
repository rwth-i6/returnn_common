Here we keep raw network helpers which create a net dict more directly, or functions used for `EvalLayer`, such as our earlier SpecAugment implementation, or RNN-T loss function wrappers, etc.

Note that almost everything in here can be considered as deprecated!
Everything can be implemented more cleanly using the `nn` framework.
If we need to wrap native external code such as for RNN-T, we can (and should) also have such a wrapper directly in `nn`.

So, try to avoid using code from here, and also do not extend this further.

However, we try to keep the behavior as stable for all (used) code in here, such that setups using these by directly importing from returnn-common should not break in the future.
Note that this was not such a hard requirement before because earlier, returnn-common was recommended to be used only versioned, e.g. via the RETURNN `import_` mechanism.
You can still use that.
