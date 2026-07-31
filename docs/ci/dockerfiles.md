# Docker images

## Custom docker images and handle_docker action

To optimize the time required to install dependencies in workflows and to make local reproduction of workflow steps easier,
we create custom Docker images for different types of validation and use them in our workflows.
The dockerfiles for these images are stored in this repository in [`.github/dockerfiles`](./../../.github/dockerfiles)
folder. Dockerfiles are organized as follows:

* [`.github/dockerfiles/nncf`](./../../.github/dockerfiles/nncf) - contains images used in nncf workflows

The changes to these dockerfiles are getting checked and applied automatically in pre-commits via the reusable
[handle_docker](https://github.com/openvinotoolkit/openvino/tree/master/.github/actions/handle_docker) action from the
OpenVINO repository, which is executed in workflows before starting actual validation in a separate job called `docker`.
The action checks if a pull request changes either dockerfiles or environment setup scripts, and if so - triggers
affected docker images build and validation with the updated images. The images are tagged with the ID of the pull
request that changes them, and the tag must be updated in git by changing PR ID in
[.github/dockerfiles/docker_tag](./../../.github/dockerfiles/docker_tag). The action will prompt you to do that once
you change something that alters docker environment.

The `changed_components` input expected by the action is produced by the `check_changes` job. This job diffs the pull request against its base
branch and emits a JSON object of the form `{"docker_env": <bool>, "dockerfiles": <bool>}`, where `docker_env` is `true`
if `.dockerignore` or any file under `.github/dockerfiles` changed, and `dockerfiles` is `true` if any file under
`.github/dockerfiles` changed.

**Important**: If you add a new environment configuration script to be used in dockerfiles, please, exclude the path to this script
from [.dockerignore](./../../.github/dockerfiles/.dockerignore) (this will make sure that Docker itself detects a script file).

The action accepts a list of the desired images to build as an input and outputs fully-qualified Docker image references
to use in workflow jobs.

### Using custom images in workflow jobs

* Make sure that the `docker` job is called in your workflow. Pass a path or multiple paths to the folders with
dockerfiles, that are going to be used further in a workflow, to `images` parameter of the `handle_docker` action.

Example, taken from [call_precommit.yml](./../../.github/workflows/call_precommit.yml):

```yaml
  docker:
    needs: check_changes
    name: Docker
    runs-on: aks-linux-4-cores-16gb-docker-build
    container:
      image: openvinogithubactions.azurecr.io/docker_build:0.2
      volumes:
        - /mount:/mount
    outputs:
      images: "${{ steps.handle_docker.outputs.images }}"
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0 # v7.0.0

      - uses: openvinotoolkit/openvino/.github/actions/handle_docker@1f6bb04df50d71dd7e392537eee6b834312f269a
        id: handle_docker
        with:
          images: |
            nncf/pytorch_cuda
          registry: 'openvinogithubactions.azurecr.io'
          dockerfiles_root_dir: '.github/dockerfiles'
          changed_components: ${{ needs.check_changes.outputs.changed_components }}
```

* Add `docker` to the `needs:` block of the job that will be executed with the desired custom image and set
`container.image` key in this job to point to the docker image taken from `handle_docker`'s outputs, like that:

```yaml
  pytorch-cuda:
    needs: docker
    ...
    container:
      image: ${{ fromJSON(needs.docker.outputs.images).nncf.pytorch_cuda }}
```

